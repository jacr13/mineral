"""Differentiable one-step simulator maps for MENTOR.

MENTOR queries the simulator from arbitrary observed states: ``step(obs, a) -> next_obs``.
Each environment needs a map from its (action-stripped) observation to the simulator state
(joint_q, joint_qd) and back. The maps below are exact for the DFlex Hopper and, up to the
unobserved horizontal position (set to 0; the dynamics on flat ground do not depend on it),
for the DFlex Ant, Humanoid and SNU humanoid with ``no_env_offset=True``.

Ant and Humanoid observations end with the previous action. Those coordinates are stripped
(``obs[:, :obs_dim]``) so that nothing about the demonstrator's actions is read.
"""

import inspect

import torch

from ...envs.dflex import DEFAULT_DFLEXENVS_KWARGS


class SimCounter:
    """Simulator transitions: ``fwd`` without a backward pass, ``fwdbwd`` with one."""

    def __init__(self):
        self.fwd = 0
        self.fwdbwd = 0

    @property
    def total(self):
        return self.fwd + self.fwdbwd

    def state_dict(self):
        return {'fwd': self.fwd, 'fwdbwd': self.fwdbwd}

    def load_state_dict(self, d):
        self.fwd, self.fwdbwd = int(d['fwd']), int(d['fwdbwd'])


class EnvSpec:
    env_name = None
    obs_dim = None  # observation coordinates that describe the state (actions stripped)
    act_dim = None
    act_offset = None  # first actuated dof in joint_act

    def make_env(self, num_envs, device, seed=0, stochastic_init=False, early_termination=False):
        import dflex.envs as DFlexEnvs

        kwargs = dict(DEFAULT_DFLEXENVS_KWARGS[self.env_name])
        env_fn = getattr(DFlexEnvs, kwargs.pop('env_name'))
        kwargs.pop('episode_length', None)
        params = inspect.signature(env_fn).parameters
        extra = {}
        if 'early_termination' in params:
            extra['early_termination'] = early_termination
        if 'no_env_offset' in params:
            extra['no_env_offset'] = True
        return env_fn(
            render=False,
            device=device,
            num_envs=num_envs,
            seed=seed,
            episode_length=1000,
            no_grad=False,
            stochastic_init=stochastic_init,
            **kwargs,
            **extra,
        )

    def act_scale(self, env, n):
        raise NotImplementedError

    def apply_action(self, env, state, a, n, num_qd):
        """Torque actuation: joint torques act_scale * clip(a) on the actuated dofs."""
        act = torch.zeros(n, num_qd, device=a.device, dtype=a.dtype)
        act[:, self.act_offset :] = torch.clip(a, -1, 1) * self.act_scale(env, n)
        state.joint_act = act.reshape(-1).contiguous()

    def to_state(self, env, obs):
        raise NotImplementedError

    def from_state(self, env, q, qd):
        raise NotImplementedError


class HopperSpec(EnvSpec):
    env_name, obs_dim, act_dim, act_offset = 'hopper', 11, 3, 3

    def act_scale(self, env, n):
        return env.action_strength

    def to_state(self, env, obs):
        n = obs.shape[0]
        q = torch.cat([torch.zeros(n, 1, device=obs.device, dtype=obs.dtype), obs[:, :5]], 1)
        return q, obs[:, 5:11]

    def from_state(self, env, q, qd):
        return torch.cat([q[:, 1:], qd], -1)


class AntSpec(EnvSpec):
    env_name, obs_dim, act_dim, act_offset = 'ant', 29, 8, 6
    num_joints = 8

    def act_scale(self, env, n):
        return env.action_strength

    def to_state(self, env, obs):
        nj = self.num_joints
        z = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        y = obs[:, 0]
        q = torch.cat([z[:, None], y[:, None], z[:, None], obs[:, 1:5], obs[:, 11 : 11 + nj]], 1)
        lin, ang = obs[:, 5:8], obs[:, 8:11]
        # obs holds the COM linear velocity; the generalized velocity is v - p x w with p = (0, y, 0)
        cross = torch.stack([y * ang[:, 2], z, -y * ang[:, 0]], 1)
        qd = torch.cat([ang, lin + cross, obs[:, 11 + nj : 11 + 2 * nj] / 0.1], 1)
        return q, qd

    def from_state(self, env, q, qd):
        from dflex.envs import ant as ant_module

        tu = ant_module.tu
        n = q.shape[0]
        pos, rot = q[:, 0:3], q[:, 3:7]
        ang = qd[:, 0:3]
        lin = qd[:, 3:6] - torch.cross(pos, ang, dim=-1)
        to_target = env.targets[:n] - pos
        to_target = torch.cat([to_target[:, :1], torch.zeros_like(to_target[:, :1]), to_target[:, 2:]], 1)
        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(rot, env.inv_start_rot[:n])
        up_vec = tu.quat_rotate(torso_quat, env.basis_vec1[:n])
        heading_vec = tu.quat_rotate(torso_quat, env.basis_vec0[:n])
        return torch.cat(
            [
                pos[:, 1:2],
                rot,
                lin,
                ang,
                q[:, 7:],
                0.1 * qd[:, 6:],
                up_vec[:, 1:2],
                (heading_vec * target_dirs).sum(-1, keepdim=True),
            ],
            1,
        )


class HumanoidSpec(AntSpec):
    env_name, obs_dim, act_dim, act_offset = 'humanoid', 55, 21, 6
    num_joints = 21

    def act_scale(self, env, n):
        return env.motor_scale * env.motor_strengths[:n]


class SNUHumanoidSpec(AntSpec):
    """Muscle-driven lower-body humanoid: 152 muscle actions for 18 actuated dofs (over-actuated).

    Like the env's step(): activation = (clip(a) * 0.5 + 0.5) * muscle_strengths, set directly at
    every step, so there is no hidden activation state. The 53 observations hold no action.
    joint_q has 22 coordinates after the root (ball joints), joint_qd 18.
    """

    env_name, obs_dim, act_dim, act_offset = 'snu_humanoid', 53, 152, 6

    def apply_action(self, env, state, a, n, num_qd):
        state.joint_act = torch.zeros(n * num_qd, device=a.device, dtype=a.dtype)
        activation = torch.clip(a, -1, 1) * 0.5 + 0.5
        env.model.muscle_activation = (activation.reshape(-1) * env.muscle_strengths[: n * self.act_dim]).contiguous()

    def to_state(self, env, obs):
        z = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        y = obs[:, 0]
        q = torch.cat([z[:, None], y[:, None], z[:, None], obs[:, 1:5], obs[:, 11:33]], 1)
        lin, ang = obs[:, 5:8], obs[:, 8:11]
        cross = torch.stack([y * ang[:, 2], z, -y * ang[:, 0]], 1)
        qd = torch.cat([ang, lin + cross, obs[:, 33:51] / env.joint_vel_obs_scaling], 1)
        return q, qd

    def from_state(self, env, q, qd):
        from dflex.envs import snu_humanoid as snu_module

        tu = snu_module.tu
        n = q.shape[0]
        pos, rot = q[:, 0:3], q[:, 3:7]
        ang = qd[:, 0:3]
        lin = qd[:, 3:6] - torch.cross(pos, ang, dim=-1)
        to_target = env.targets[:n] + env.start_pos[:n] - pos
        to_target = torch.cat([to_target[:, :1], torch.zeros_like(to_target[:, :1]), to_target[:, 2:]], 1)
        target_dirs = tu.normalize(to_target)
        torso_quat = tu.quat_mul(rot, env.inv_start_rot[:n])
        up_vec = tu.quat_rotate(torso_quat, env.basis_vec1[:n])
        heading_vec = tu.quat_rotate(torso_quat, env.basis_vec0[:n])
        return torch.cat(
            [
                pos[:, 1:2],
                rot,
                lin,
                ang,
                q[:, 7:],
                env.joint_vel_obs_scaling * qd[:, 6:],
                up_vec[:, 1:2],
                (heading_vec * target_dirs).sum(-1, keepdim=True),
            ],
            1,
        )


class SynergySpec(EnvSpec):
    """Actions in a low-dimensional synergy space z in [-1, 1]^d.

    Decoded to the base env's actions as a = clip(mean + (z * scale) @ basis.T, -1, 1), with a PCA
    basis of recovered muscle activations. Everything else is delegated to the base spec.
    """

    def __init__(self, base, mean, basis, scale):
        self.base, self.mean, self.basis, self.scale = base, mean, basis, scale
        self.env_name, self.obs_dim, self.act_offset = base.env_name, base.obs_dim, base.act_offset
        self.act_dim = basis.shape[1]

    def decode(self, z):
        return (self.mean + (torch.clip(z, -1, 1) * self.scale) @ self.basis.T).clamp(-1, 1)

    def encode(self, a):
        return (((a - self.mean) @ self.basis) / self.scale).clamp(-1, 1)

    def make_env(self, *args, **kwargs):
        return self.base.make_env(*args, **kwargs)

    def apply_action(self, env, state, z, n, num_qd):
        self.base.apply_action(env, state, self.decode(z), n, num_qd)

    def to_state(self, env, obs):
        return self.base.to_state(env, obs)

    def from_state(self, env, q, qd):
        return self.base.from_state(env, q, qd)


SPECS = {'hopper': HopperSpec, 'ant': AntSpec, 'humanoid': HumanoidSpec, 'snu_humanoid': SNUHumanoidSpec}


def get_spec(env_name):
    if env_name not in SPECS:
        raise ValueError(f'MENTOR has no observation-to-state map for env {env_name!r}; supported: {sorted(SPECS)}')
    return SPECS[env_name]()


class OneStep:
    """``obs, a -> next_obs`` for a fixed batch of ``n`` states, differentiable in both arguments.

    Owns its own DFlex env (``n`` parallel copies) and only uses its model and integrator.
    A step with ``grad=True`` records the DFlex tape: always back-propagate through it,
    otherwise the tape leaks (use ``grad=False`` for forward-only queries).
    """

    def __init__(self, spec, n, device, counter):
        self.spec, self.n, self.device, self.counter = spec, n, device, counter
        self.env = spec.make_env(n, device)
        self.num_q = self.env.num_joint_q
        self.num_qd = self.env.num_joint_qd

    def __call__(self, obs, a, grad=True):
        if obs.shape[0] != self.n:
            raise ValueError(f'OneStep built for batch {self.n}, got {obs.shape[0]}')
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            return self._step(obs, a, grad)

    def _step(self, obs, a, grad):
        spec, env, n = self.spec, self.env, self.n
        state = env.model.state()
        q, qd = spec.to_state(env, obs)
        state.joint_q = q.reshape(-1).contiguous()
        state.joint_qd = qd.reshape(-1).contiguous()
        spec.apply_action(env, state, a, n, self.num_qd)
        out = env.integrator.forward(env.model, state, env.sim_dt, env.sim_substeps, env.MM_caching_frequency)
        if grad:
            self.counter.fwdbwd += n
        else:
            self.counter.fwd += n
        return spec.from_state(env, out.joint_q.view(n, -1), out.joint_qd.view(n, -1))
