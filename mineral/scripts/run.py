import functools
import os
import pprint
import sys

from ..common.quiet import quiet_third_party

quiet_third_party()  # before Hydra and gym are imported

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from hydra.utils import to_absolute_path  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from termcolor import cprint  # noqa: E402

from ..common.job_clock import JobClock  # noqa: E402
from .utils import make_run_names  # noqa: E402

load_dotenv()


def make_envs(config):
    from .. import envs

    task_suite = config.task.get('suite', 'isaacgymenvs')
    TaskSuite = getattr(envs, task_suite)
    return TaskSuite.make_envs(config)


def make_datasets(config, env):
    if not hasattr(config.agent, 'datasets'):
        return None

    from .. import envs

    task_suite = config.task.get('suite', 'isaacgymenvs')
    TaskSuite = getattr(envs, task_suite)
    return TaskSuite.make_datasets(config, env)


def save_run_metadata(logdir, resolved_config, run_name, run_id, project=None, run_group=None):
    run_metadata = {
        'logdir': logdir,
        'run_name': run_name,
        'run_id': run_id,
        'project': project,
        'run_group': run_group,
        'cluster_run_id': os.getenv('CLUSTER_RUN_ID'),
        'slurm_job_id': os.getenv('SLURM_JOB_ID'),
        'git_commit': os.getenv('MINERAL_GIT_COMMIT'),
    }
    yaml.dump(run_metadata, open(os.path.join(logdir, 'run_metadata.yaml'), 'w'), default_flow_style=False)
    yaml.dump(resolved_config, open(os.path.join(logdir, 'resolved_config.yaml'), 'w'), default_flow_style=False)


def main(config: DictConfig):
    if 'isaacgym' in sys.modules or 'isaacgymenvs' in sys.modules:
        from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    else:
        from .utils import set_np_formatting, set_seed

    # set numpy formatting for printing only
    set_np_formatting()

    from .utils import limit_threads

    limit_threads(1)

    import torch

    if not torch.cuda.is_available():
        config.device_id = -1

    assert config.seed >= 0
    set_seed = functools.partial(set_seed, torch_deterministic=config.torch_deterministic)
    set_seed(0)

    # --- Setup Run ---
    logdir = config.logdir
    os.makedirs(logdir, exist_ok=True)

    if config.ckpt:
        config.ckpt = to_absolute_path(config.ckpt)

    if config.multi_gpu:
        from accelerate import Accelerator

        accelerator = Accelerator()
        rank = int(os.getenv('LOCAL_RANK', '0'))
        if str(accelerator.device) == 'cuda':
            pass
        else:
            assert rank == accelerator.device.index, print(rank, accelerator.device, accelerator.device.index)

        config.sim_device = f'cuda:{rank}'
        config.rl_device = f'cuda:{rank}'
        config.graphics_device_id = rank

        # if rank != 0:
        #     f = open(os.path.join(config.logdir, 'log_{rank}.txt', 'w'))
        #     sys.stdout = f
    else:
        accelerator = None
        rank = 0

        # use the same device for sim and rl
        config.sim_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
        config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
        config.graphics_device_id = config.device_id if config.device_id >= 0 else 0

    resolved_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print(pprint.pformat(resolved_config, compact=True, indent=1), '\n')

    if rank == 0:
        job_id = os.environ.get("SLURM_JOB_ID")
        print(f'SLURM_JOB_ID: {job_id}')
        resolved_config['slurm_job_id'] = job_id
        resolved_config['cluster_run_id'] = os.getenv('CLUSTER_RUN_ID')
        resolved_config['git_commit'] = os.getenv('MINERAL_GIT_COMMIT')

        names = make_run_names(config, resolved_config, logdir)
        print(f"run_group: {names['run_group']}, run_name: {names['run_name']}, run_id: {names['run_id']}")
        save_run_metadata(logdir, resolved_config, **names)

    # generate random seeds (deterministically given config.seed)
    # should be same across workers since each worker should add its global rank
    to_seed = ['setup', 'train', 'eval', 'env_train', 'env_eval']
    rng = np.random.RandomState(config.seed)
    seeds = rng.randint(0, int(1e6), len(to_seed))
    seeds = {k: int(seeds[i]) for i, k in enumerate(to_seed)}
    cprint(f'Base Seed: {config.seed}, Seeds: {seeds}', 'green', attrs=['bold'])
    set_seed(seeds['setup'] + rank)

    # --- Make Envs, Datasets, Agent ---
    cprint('Making Envs', 'green', attrs=['bold'])
    env = make_envs(config)
    print('-' * 20)
    print(f'Env: {env}')

    datasets = make_datasets(config, env)
    print(f'Datasets: {datasets}')

    from .. import agents

    AgentCls = getattr(agents, config.agent.algo)
    print(f'AgentCls: {AgentCls}', '\n')

    job_clock = JobClock(config.get("max_runtime"))
    job_clock.arm()
    job_clock.print_info()
    agent = AgentCls(config, logdir=logdir, accelerator=accelerator, datasets=datasets, env=env, job_clock=job_clock)

    if config.ckpt:
        print(f'Loading checkpoint: {config.ckpt}')
        agent.load(config.ckpt, ckpt_keys=config.ckpt_keys)

    # --- Run Agent ---
    print('-' * 20)
    cprint(f'Running: {config.run}', 'green', attrs=['bold'])

    try:
        if config.run == 'train':
            set_seed(seeds['train'] + rank)
            agent.train()
        elif config.run == 'eval':
            set_seed(seeds['eval'] + rank)
            agent.eval()
        elif config.run == 'train_eval':
            set_seed(seeds['train'] + rank)
            agent.train()

            agent.load(os.path.join(agent.ckpt_dir, 'final.pth'))

            set_seed(seeds['eval'] + rank)
            agent.eval()
        else:
            raise NotImplementedError(config.run)
    except (KeyboardInterrupt, Exception) as e:
        # raise e

        import traceback

        traceback.print_exc()
        print(f'Exception: {e}')
        agent.writer.record_exception(e)  # info.json: status failed/interrupted + traceback

    # --- Cleanup ---
    if hasattr(env, 'renderer') and config.task.env.render:
        env.renderer.save()

    agent.writer.close()  # flushes every logger and finishes their runs
    # env.close()


if __name__ == '__main__':
    c = []
    hydra.main(
        config_name='config',
        config_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../cfgs'),
        version_base='1.3',
    )(lambda x: c.append(x))()
    config = c[0]

    task_suite = config.task.get('suite', 'isaacgymenvs')
    if task_suite == 'isaacgymenvs':
        from ..envs.isaacgymenvs import import_isaacgym

        import_isaacgym()  # (need to import before torch)

    main(config)
