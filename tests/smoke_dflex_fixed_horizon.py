"""Run manually in the project Docker image with --gpus=all."""

import importlib.util
from pathlib import Path

import torch
from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location('adapter', Path(__file__).resolve().parents[1] / 'mineral/envs/dflex.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(), flush=True)
for name in ('ant', 'hopper', 'humanoid', 'snu_humanoid'):
    cfg = OmegaConf.create({'task': {'env': {'env_name': name, 'numEnvs': 2, 'early_termination': False, 'no_grad': True, 'stochastic_init': False, 'render': False, 'no_env_offset': True}}, 'sim_device': 'cuda:0', 'seed': 100})
    print('Constructing', name, flush=True)
    env = adapter.make_envs(cfg)
    obs = env.reset()
    fell = torch.zeros(2, dtype=torch.bool, device='cuda:0')
    for step in range(1000):
        action = torch.zeros((2, env.num_actions), device='cuda:0')
        obs, reward, done, info = env.step(action)
        fell |= obs[:, 0] < env.termination_height
        assert torch.isfinite(obs).all(), (name, step, 'invalid observations')
        assert torch.isfinite(reward).all(), (name, step, 'invalid rewards')
        if step < 999:
            assert not done.any(), (name, step, 'early reset')
            assert (env.progress_buf == step + 1).all(), (name, step, 'progress reset')
        else:
            assert done.all(), (name, 'missing horizon reset')
        if (step + 1) % 250 == 0:
            print(name, 'step', step + 1, 'fell', fell.tolist(), flush=True)
    print('PASS', name, '1000 GPU steps; no early reset; horizon reset works', flush=True)
    del env
    torch.cuda.empty_cache()
