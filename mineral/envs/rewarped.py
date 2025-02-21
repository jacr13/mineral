import re
import importlib

from omegaconf import OmegaConf


def make_envs(config):
    env = config.task.env
    env_kwargs = OmegaConf.to_container(env, resolve=True)
    env_name, num_envs = env_kwargs.pop("env_name"), env_kwargs.pop("numEnvs")
    env_suite = env_kwargs.pop("env_suite")

    def camelcase_to_snakecase(str):
        s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

    ENV = importlib.import_module(f"rewarped.envs.{env_suite}.{camelcase_to_snakecase(env_name)}")
    env_fn = getattr(ENV, env_name)
    env = env_fn(
        num_envs=num_envs,
        device=config.sim_device,
        seed=config.seed,
        **env_kwargs,
    )

    return env
