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
    # The pinned DFlex humanoids hard-code fall resets in calculateReward;
    # unlike Ant/Hopper, their constructors have no early_termination option.
    if (
        env_name in ('HumanoidEnv', 'SNUHumanoidEnv')
        and 'early_termination' in env_kwargs
        and 'early_termination' not in inspect.signature(env_fn).parameters
    ):
        if not env_kwargs.pop('early_termination'):
            env_fn = type(f'FixedHorizon{env_name}', (_FixedHorizonHumanoid, env_fn), {})
    env = env_fn(
        num_envs=num_envs,
        device=config.sim_device,
        seed=config.seed,
        **env_kwargs,
    )

    return env
