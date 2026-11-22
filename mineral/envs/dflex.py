import inspect

import torch
from omegaconf import OmegaConf

DEFAULT_DFLEXENVS_KWARGS = {
    'ant': {'env_name': 'AntEnv', 'episode_length': 1000, 'MM_caching_frequency': 16},
    'anymal': {'env_name': 'AnymalEnv', 'episode_length': 1000, 'MM_caching_frequency': 16},
    'cartpoleswingup': {'env_name': 'CartPoleSwingUpEnv', 'episode_length': 240, 'MM_caching_frequency': 4},
    'cheetah': {'env_name': 'CheetahEnv', 'episode_length': 1000, 'MM_caching_frequency': 16},
    'hopper': {'env_name': 'HopperEnv', 'episode_length': 1000, 'MM_caching_frequency': 16},
    'humanoid': {'env_name': 'HumanoidEnv', 'episode_length': 1000, 'MM_caching_frequency': 48},
    'snu_humanoid': {'env_name': 'SNUHumanoidEnv', 'episode_length': 1000, 'MM_caching_frequency': 8},
}


class _FixedHorizonInfo:
    """Expose terminal signals and observations without resetting on falls."""

    def calculateReward(self):
        super().calculateReward()
        self._terminal_obs = self.obs_buf.clone()
        self._terminal = (self.obs_buf[:, 0] < self.termination_height) | self.reset_buf.bool()

    def step(self, actions):
        obs, reward, done, info = super().step(actions)
        return obs, reward, done, {
            **info,
            'termination': self._terminal,
            'obs_before_reset': self._terminal_obs,
        }


class _FixedHorizonHumanoid:
    """Disable fall resets in legacy DFlex humanoids without changing rewards."""

    def calculateReward(self):
        super().calculateReward()
        joint_q = self.state.joint_q.view(self.num_envs, -1)
        joint_qd = self.state.joint_qd.view(self.num_envs, -1)
        invalid = (
            ~torch.isfinite(self.obs_buf).all(dim=-1)
            | ~torch.isfinite(joint_q).all(dim=-1)
            | ~torch.isfinite(joint_qd).all(dim=-1)
            | (joint_q.abs() > 1e6).any(dim=-1)
            | (joint_qd.abs() > 1e6).any(dim=-1)
        )
        self.reset_buf = (invalid | (self.progress_buf >= self.episode_length)).to(self.reset_buf.dtype)


def _release_dangling_sim_nodes(env):
    """DFlex Anymal leaks GPU memory (~0.46 GB per 32-step window at 16 envs, ~2 GB at 64; 2026-11-22).

    DFlex's ``SimulateFunc`` keeps each step's tape, inputs and outputs on its autograd node and frees them only in
    its backward (``ctx.tape.reset()``); node and outputs form a reference cycle otherwise. In AnymalEnv the reward is
    computed from the observation *before* the step, so the last simulation step of a window can stay off the
    backward path; its node then keeps the whole window's tapes alive, and every window adds to that. Fix: when a new
    trajectory starts (``clear_grad``), release the previous window's last node (already used or never usable: the
    graph is cut there anyway), so gradients are unchanged. Applied to AnymalEnv only.
    """
    orig = env.clear_grad

    def release(roots):
        # walk the finished window's graph (from the env's state / observation tensors; an auto-reset in the last
        # step puts copy nodes in front of the simulation node) and release every simulation node reached
        seen, stack = set(), [getattr(t, 'grad_fn', None) for t in roots if torch.is_tensor(t)]
        while stack:
            gf = stack.pop()
            if gf is None or id(gf) in seen:
                continue
            seen.add(id(gf))
            if type(gf).__name__ == 'SimulateFuncBackward':
                gf.tape.reset()
                gf.inputs = ()
                gf.outputs = []
            stack.extend(f for f, _ in getattr(gf, 'next_functions', ()))

    def clear_grad(checkpoint=None):
        release([env.state.joint_q, env.state.joint_qd, env.state.joint_act, getattr(env, 'obs_buf', None)])
        return orig(checkpoint)

    env.clear_grad = clear_grad


def _reset_returns_obs(env):
    """AnymalEnv.reset() returns (obs, extras); every other DFlex env, and the agents, expect the observation alone
    (otherwise the evaluation gets {'obs': (obs, extras)} and the policy NaN actions). Its internal reset(env_ids)
    calls ignore the return value."""
    reset = env.reset

    def reset_obs(*args, **kwargs):
        out = reset(*args, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    env.reset = reset_obs


def _terminate_any_nonfinite(env):
    """AnymalEnv.compute_termination is meant to reset environments whose simulation blew up, but its mask
    ``~(isfinite(x).sum(-1) > 0)`` only flags an environment when *every* entry is non-finite; a partly-NaN state
    (seen in the evaluation after ~10 steps of an untrained policy) is then not reset and reaches the policy. Also
    terminate environments with any non-finite observation or joint state; step() then resets them as usual."""
    orig = env.compute_termination

    def compute_termination(obs, act):
        term = orig(obs, act)
        q = env.state.joint_q.view(env.num_envs, -1)
        qd = env.state.joint_qd.view(env.num_envs, -1)
        bad = ~(torch.isfinite(obs).all(-1) & torch.isfinite(q).all(-1) & torch.isfinite(qd).all(-1))
        return term | bad

    env.compute_termination = compute_termination


def make_envs(config):
    env = config.task.env
    env_kwargs = OmegaConf.to_container(env, resolve=True)
    env_name, num_envs = env_kwargs.pop('env_name'), env_kwargs.pop('numEnvs')

    env_kwargs = {
        **DEFAULT_DFLEXENVS_KWARGS[env_name],
        **env_kwargs,
    }
    env_name = env_kwargs.pop('env_name')

    # DIFFRL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../third_party/DiffRL'))
    # sys.path.append(DIFFRL_PATH)
    # import envs as DFlexEnvs

    import dflex.envs as DFlexEnvs

    env_fn = getattr(DFlexEnvs, env_name)
    fixed_horizon = env_kwargs.get('early_termination') is False
    # The pinned DFlex humanoids hard-code fall resets in calculateReward;
    # unlike Ant/Hopper, their constructors have no early_termination option.
    if (
        env_name in ('HumanoidEnv', 'SNUHumanoidEnv')
        and 'early_termination' in env_kwargs
        and 'early_termination' not in inspect.signature(env_fn).parameters
    ):
        if not env_kwargs.pop('early_termination'):
            env_fn = type(f'FixedHorizon{env_name}', (_FixedHorizonHumanoid, env_fn), {})
    if fixed_horizon and env_name in ('AntEnv', 'HopperEnv', 'HumanoidEnv', 'SNUHumanoidEnv'):
        env_fn = type(f'FixedHorizonInfo{env_name}', (_FixedHorizonInfo, env_fn), {})
    env = env_fn(
        num_envs=num_envs,
        device=config.sim_device,
        seed=config.seed,
        **env_kwargs,
    )
    _keep_obs_before_reset(env)
    if env_name == 'AnymalEnv':
        _release_dangling_sim_nodes(env)
        _reset_returns_obs(env)
        _terminate_any_nonfinite(env)
    return env


def _keep_obs_before_reset(env):
    """Make no_grad DFlex envs return infos['obs_before_reset'].

    With no_grad=True the pinned DFlex envs do not return it, so the observation of the last step of an episode is
    lost to the auto-reset. step() sets ``extras`` and then calls ``self.reset(env_ids)``: wrap the instance's reset
    to store the observation buffer first (as DFlex does when no_grad=False).
    """
    if not getattr(env, 'no_grad', False):
        return
    reset = env.reset

    def reset_keeping_obs(env_ids=None, force_reset=True):
        extras = getattr(env, 'extras', None)
        if env_ids is not None and isinstance(extras, dict) and 'obs_before_reset' not in extras:
            extras['obs_before_reset'] = env.obs_buf.clone()
        # Resetting given envs: no flag. The other DFlex envs are reset(env_ids, force_reset=True) and only use the
        # flag when env_ids is None (unchanged); AnymalEnv.reset(env_ids, grads, force_reset) would take a positional
        # flag as `grads` (skipping the reset) and, given force_reset, reset every env.
        if env_ids is not None:
            return reset(env_ids)
        return reset(env_ids, force_reset=force_reset)

    env.reset = reset_keeping_obs
