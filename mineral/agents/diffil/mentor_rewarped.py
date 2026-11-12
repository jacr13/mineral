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
        self.obs_keys = [k for k in obs_keys.split('|') if k] if isinstance(obs_keys, str) else list(obs_keys)
        self.stage1 = stage1
        self.device = device
        probe = self.make_env(1, device, grad=False)
        obs = probe.reset()
        self.state_names = tuple(type(probe).state_tensors_names)  # a no-grad env clears the instance's copy
        self.act_dim = int(probe.num_actions)
        self.action_scale = float(getattr(probe, 'action_scale', 1.0))
        self.slices, start = {}, 0
        for k in self.obs_keys:
            size = obs[k].reshape(1, -1).shape[1]
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
        return torch.cat([obs[k].reshape(n, -1) for k in self.obs_keys], 1)

    # ---- full state ----
    def get_states(self, env, ids=None):
        """Full simulator state of envs ``ids`` (all by default): {name: [len(ids), ...]}, detached copies."""
        n = env.num_envs
        out = {}
        for k in self.state_names:
            v = getattr(env.state, k).detach()
            v = v.view(n, -1, *v.shape[1:])
            out[k] = (v if ids is None else v[ids]).clone()
        return out

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

    def __init__(self, spec, n, device, counter):
        self.spec, self.n, self.device, self.counter = spec, n, device, counter
        self.env = spec.make_env(n, device, grad=True)
        self.env.reset()

    def load(self, states):
        if next(iter(states.values())).shape[0] != self.n:
            raise ValueError(f'StateStep built for batch {self.n}, got {next(iter(states.values())).shape[0]}')
        for k, v in states.items():
            self.env.state.assign(k, v.reshape(-1, *v.shape[2:]).clone().contiguous())

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
