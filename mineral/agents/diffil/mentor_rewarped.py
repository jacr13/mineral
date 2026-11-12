"""Full-state MENTOR for Rewarped tasks (e.g. DexDeform Flip, an MPM soft body).

The observation of these tasks does not determine the simulator state (MPM particles carry hidden deformation
gradients, stresses and affine velocity fields), so there is no observation-to-state map. MENTOR only needs one:
the teacher plans from states the *learner* visited, and those states are simulated, hence fully known. The spec
below saves and restores the full Rewarped state of a batch of environments; the teacher starts its plans from
those saved states and compares observations with the demonstrations, exactly as in the torque-driven case.

Stage I is task-specific: for kinematic, velocity-controlled hands (DexDeform), the action is the joint
displacement, a_t = (q_{t+1} - q_t) / action_scale, which is read off the observations without simulation.
"""

import torch
from omegaconf import OmegaConf

from .mentor_sim import OneStep


class RewarpedSpec:
    """Observation vector, full-state save/restore and Stage I for a Rewarped env built as ``task=Rewarped`` does.

    ``obs_keys``: observation-dict keys concatenated (flattened per env) into MENTOR's observation vector.
    ``stage1``: 'joint_velocity' (a = delta joint_q / action_scale, clipped to [-1, 1]).
    """

    full_state = True
    act_offset = 0

    def __init__(self, full_cfg, obs_keys, stage1='joint_velocity', device='cuda:0'):
        self.env_cfg = OmegaConf.to_container(full_cfg.task.env, resolve=True)
        self.env_name = self.env_cfg['env_name']
        keys = [k for k in obs_keys.split('|') if k] if isinstance(obs_keys, str) else list(obs_keys)
        self.obs_keys, self.key_slices = [], {}
        for k in keys:  # 'name' or 'name[a:b]' (a slice of the flattened entry, e.g. obs[:56] drops the last action)
            name, _, rest = k.partition('[')
            sl = slice(None)
            if rest:
                a, _, b = rest.rstrip(']').partition(':')
                sl = slice(int(a) if a else None, int(b) if b else None)
            self.obs_keys.append(name)
            self.key_slices[name] = sl
        self.stage1 = stage1
        self.adapter = ADAPTERS.get(self.env_name, Adapter)()
        self.device = device
        probe = self.make_env(1, device, grad=False)
        obs = probe.reset()
        self.state_names = tuple(type(probe).state_tensors_names)  # a no-grad env clears the instance's copy
        self.act_dim = int(probe.num_actions)
        self.action_scale = float(getattr(probe, 'action_scale', 1.0))
        self.slices, start = {}, 0
        for k in self.obs_keys:
            size = obs[k].reshape(1, -1)[:, self.key_slices[k]].shape[1]
            self.slices[k] = slice(start, start + size)
            start += size
        self.obs_dim = start
        self.episode_length = int(probe.episode_length)
        del probe

    # ---- environments ----
    def make_env(self, num_envs, device, seed=0, grad=True, **_):
        from mineral.envs.rewarped import make_envs

        env_cfg = dict(self.env_cfg)
        env_cfg.update(numEnvs=int(num_envs), no_grad=not grad, render=False)
        cfg = OmegaConf.create({'sim_device': device, 'seed': int(seed), 'task': {'env': env_cfg}})
        # Graph capture stays on: assign() copies into the captured buffers, and it keeps the tape memory per
        # step instead of per substep (MPM substeps hold full grids).
        return make_envs(cfg)

    def obs_vec(self, obs):
        n = obs[self.obs_keys[0]].shape[0]
        return torch.cat([obs[k].reshape(n, -1)[:, self.key_slices[k]] for k in self.obs_keys], 1)

    # ---- full state ----
    def get_states(self, env, ids=None):
        """Full simulator state of envs ``ids`` (all by default): {name: [len(ids), ...]}, detached copies."""
        n = env.num_envs
        out = {}
        for k in self.state_names:
            v = getattr(env.state, k).detach()
            v = v.view(n, -1, *v.shape[1:])
            out[k] = (v if ids is None else v[ids]).clone()
        for k in self.adapter.extra:  # env variables that enter the observation (e.g. a goal)
            v = getattr(env, k).detach()
            out[k] = (v if ids is None else v[ids]).clone()
        return out

    def states_from_obs(self, env, obs):
        return self.adapter.states_from_obs(self, env, obs)

    def make_obs_step(self, n, device, counter):
        """``step(obs, a, grad)`` for Stage I inversion, when the adapter has an observation-to-state map."""
        return ObsStep(self, n, device, counter)

    @staticmethod
    def cat_states(states):
        return {k: torch.cat([s[k] for s in states]) for k in states[0]}

    @staticmethod
    def take(states, idx):
        return {k: v[idx] for k, v in states.items()}

    @staticmethod
    def repeat(states, m):
        return {k: v.repeat_interleave(m, 0) for k, v in states.items()}

    # ---- Stage I ----
    def recover_actions(self, S, S1):
        """Demo actions from observations (no simulation). S, S1: [..., obs_dim]."""
        if self.stage1 == 'invert':
            raise ValueError("stage1 'invert' runs Levenberg-Marquardt through make_obs_step(), not recover_actions()")
        if self.stage1 != 'joint_velocity':
            raise ValueError(f'unknown Rewarped Stage I {self.stage1!r}')
        if 'joint_q' not in self.slices:
            raise ValueError("stage1 'joint_velocity' needs 'joint_q' in mentor.obs_keys")
        sl = self.slices['joint_q']
        return ((S1[..., sl] - S[..., sl]) / self.action_scale).clamp(-1, 1)


class StateStep:
    """Batched differentiable stepping of a Rewarped env from saved full states (``n`` envs).

    load(states) sets the state; step(a, grad) applies one control step and returns the observation vector.
    Simulator transitions are counted like OneStep (``fwdbwd`` with a backward pass, ``fwd`` without).
    """

    def __init__(self, spec, n, device, counter, grad=True):
        self.spec, self.n, self.device, self.counter = spec, n, device, counter
        self.env = spec.make_env(n, device, grad=grad)
        self.env.reset()

    def load(self, states):
        if next(iter(states.values())).shape[0] != self.n:
            raise ValueError(f'StateStep built for batch {self.n}, got {next(iter(states.values())).shape[0]}')
        extra = self.spec.adapter.extra
        for k, v in states.items():
            if k in extra:
                getattr(self.env, k)[:] = v
            else:
                self.env.state.assign(k, v.reshape(-1, *v.shape[2:]).clone().contiguous())
        self.spec.adapter.after_load(self.env)

    def step(self, a, grad=True):
        env = self.env
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            env.pre_physics_step(a)
            env.do_physics_step()
            env.compute_observations()
            y = self.spec.obs_vec(env.obs_buf)
        if grad:
            self.counter.fwdbwd += self.n
        else:
            self.counter.fwd += self.n
        return y if grad else y.detach()


class ObsStep:
    """``step(obs, a, grad) -> next obs`` through Rewarped, via the adapter's observation-to-state map (Stage I)."""

    def __init__(self, spec, n, device, counter):
        self.spec = spec
        self.sim = StateStep(spec, n, device, counter, grad=False)  # Levenberg-Marquardt is forward-only

    def __call__(self, obs, a, grad=False):
        self.sim.load(self.spec.states_from_obs(self.sim.env, obs))
        return self.sim.step(a, grad=grad)


class Adapter:
    """Task-specific pieces of a Rewarped env. ``extra``: env attributes saved and restored with the state."""

    extra = ()

    def states_from_obs(self, spec, env, obs):
        raise ValueError(f'no observation-to-state map for {spec.env_name}; use a Stage I that needs none')

    def after_load(self, env):
        pass


class AllegroHandAdapter(Adapter):
    """IsaacGymEnvs AllegroHand (Rewarped, observationType full_state without forces): exact state map.

    obs[0:16] hand joints (unscaled to [-1, 1]), [16:32] hand joint velocities (x vel_obs_scale), [32:35] cube
    position, [35:39] cube quaternion, [39:42] cube linear velocity (COM), [42:45] angular velocity (x vel_obs_scale),
    [45:52] goal pose, [52:56] cube-goal rotation, [56:72] previous action (not used: obs[:56]).
    """

    extra = ('goal_states',)

    def states_from_obs(self, spec, env, obs):
        from rewarped.envs.isaacgymenvs.allegro_hand import scale

        n = obs.shape[0]
        lo = env.hand_dof_lower_limits.reshape(-1, 16)[:1]
        hi = env.hand_dof_upper_limits.reshape(-1, 16)[:1]
        vs = env.vel_obs_scale
        pos = obs[:, 32:35] + env.env_offsets.reshape(-1, 3)[:n]
        ang = obs[:, 42:45] / vs
        q = torch.cat([scale(obs[:, 0:16], lo, hi), pos, obs[:, 35:39]], 1)
        qd = torch.cat([obs[:, 16:32] / vs, ang, obs[:, 39:42] + torch.cross(pos, ang, dim=-1)], 1)
        out = {'joint_q': q, 'joint_qd': qd}
        if 'joint_tau' in spec.state_names:
            out['joint_tau'] = torch.zeros_like(getattr(env.state, 'joint_tau').view(env.num_envs, -1)[:n])
        goal = env.goal_states[:n].detach().clone()
        goal[:, 0:3] = obs[:, 45:48] + env.env_offsets.reshape(-1, 3)[:n]
        goal[:, 3:7] = obs[:, 48:52]
        out['goal_states'] = goal
        return out

    def after_load(self, env):
        import warp as wp

        # maximal coordinates (contacts) from the assigned joint coordinates
        wp.sim.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, None, env.state_0)


ADAPTERS = {'AllegroHand': AllegroHandAdapter}


def make_state_teacher_class(base):
    """RejoinTeacher variant whose plans start from saved full states (x = (obs, states))."""

    class StateRejoinTeacher(base):
        def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
            super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
            self.make_sim(spec, batch, device, counter)  # also when screening is off

        def make_sim(self, spec, n, device, counter):
            # One batch-sized simulator: MPM envs are large, so windows are screened sequentially on it.
            if n != self.batch:
                return None
            if getattr(self, '_sim', None) is None:
                self._sim = StateStep(spec, n, device, counter)
            return self._sim

        def rollout(self, sim, x, a, grad):
            _, states = x
            sim.load(states)
            y, outside = None, 0.0
            for k in range(self.K):
                y = sim.step(a[:, k], grad=grad)
                if self.barrier > 0:
                    outside = outside + (((self.lo - y).clamp_min(0) + (y - self.hi).clamp_min(0)) * self.w).pow(2).sum(-1)
            return y, outside

        @torch.no_grad()
        def anchored(self, sim, x, a_demo):
            if self.anchor_policy is None:
                return a_demo
            obs, states = x
            sim.load(states)
            y, acts = obs, []
            for k in range(self.K):
                a = self.anchor_policy(y).clamp(-1, 1)
                acts.append(a)
                if k < self.K - 1:
                    y = sim.step(a, grad=False)
            return (1 - self.anchor_mix) * a_demo + self.anchor_mix * torch.stack(acts, 1)

        def label(self, x):
            """x = (obs [n, D], states {name: [n, ...]}); returns labels and rejoin costs."""
            obs, states = x
            obs = obs.detach()
            n, m = obs.shape[0], self.m
            top, j = self.windows(obs)
            ar = torch.arange(n, device=self.device)
            sim = self._sim
            if self.screen:  # warm-start cost of each candidate window, one forward pass per window
                c0 = []
                with torch.no_grad():
                    for w in range(m):
                        a_w = self.a_ref(top[:, w], j[:, w])
                        c0.append(self.cost(self.rollout(sim, (obs, states), a_w, grad=False), a_w, top[:, w], j[:, w], a_w))
                sel = torch.stack(c0, 1).argmin(1)
                traj, jj = top[ar, sel], j[ar, sel]
            else:  # one window per state (m = 1)
                traj, jj = top[:, 0], j[:, 0]
            a_ref = self.a_ref(traj, jj)
            if self.anchor_policy is not None:
                a_ref = self.anchored(sim, (obs, states), a_ref)
            best_a, best = self.optimise(sim, (obs, states), traj, jj, a_ref)
            self.last_choice = ((obs, states), best_a, traj, jj)
            return best_a[:, 0], best

    return StateRejoinTeacher
