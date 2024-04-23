import os
import re
from contextlib import contextmanager

import numpy as np
import torch
from omegaconf import OmegaConf

from ..common.metrics import Metrics
from ..common.writer import TensorboardWriter, WandbWriter, Writer


class Agent:
    def __init__(self, full_cfg, logdir=None, accelerator=None, datasets=None, env=None):
        assert getattr(self, 'network_config', False)
        assert getattr(self, 'num_actors', False)

        self.full_cfg = full_cfg
        self.logdir = logdir

        # --- Device ---
        self.rank = -1
        self.device = full_cfg.rl_device
        self.multi_gpu = full_cfg.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv('LOCAL_RANK', '0'))
            self.rank_size = int(os.getenv('WORLD_SIZE', '1'))

            assert accelerator is not None
            self.accelerator = accelerator
            self.device = self.accelerator.device

        # --- Datasets ---
        self.datasets = datasets

        # --- Environment ---
        self.env = env
        action_space = self.env.action_space
        self.action_dim = action_space.shape[0]
        self.env_autoresets = full_cfg.task.get('env_autoresets', True)  # set to False to explicitly call env.reset

        # --- Inputs ---
        self.normalize_input = self.network_config.get('normalize_input', False)
        self.obs_rms_keys = re.compile(self.network_config.get('obs_rms_keys', ''))
        self.cpu_obs_keys = re.compile(self.network_config.get('cpu_obs_keys', '$^'))
        self.observation_space = self.env.observation_space
        try:
            obs_space = {k: v.shape for k, v in self.observation_space.spaces.items()}
        except:
            obs_space = {'obs': self.observation_space.shape}
        self.obs_space = obs_space

        # --- Metrics ---
        self.tracker_len = full_cfg.agent.get('tracker_len', 100)
        self.metrics_kwargs = full_cfg.agent.get('metrics_kwargs', {})
        self.env_render = full_cfg.env_render
        self.metrics = self._create_metrics(self.tracker_len, self.metrics_kwargs)

        # --- Logging ---
        self.ckpt_dir = os.path.join(self.logdir, 'ckpt')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.logdir, 'tb')
        os.makedirs(self.tb_dir, exist_ok=True)

        resolved_config = OmegaConf.to_container(full_cfg, resolve=True)
        writers = [
            WandbWriter(),
            TensorboardWriter(self.tb_dir, resolved_config),
        ]
        self.tb_summary_writer = writers[1].writer
        self.writer = Writer(writers)

        self.print_every = full_cfg.agent.get('print_every', -1)
        self.ckpt_every = full_cfg.agent.get('ckpt_every', -1)
        self.eval_every = full_cfg.agent.get('eval_every', -1)
        self.best_stat = None

        # --- Training ---
        self.epoch = -1
        self.mini_epoch = -1
        self.agent_steps = 0

    def get_actions(self, obs, sample: bool = True):
        raise NotImplementedError

    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def set_train(self):
        raise NotImplementedError

    def set_eval(self):
        raise NotImplementedError

    def save(self, f):
        raise NotImplementedError

    def load(self, f):
        raise NotImplementedError

    def _create_metrics(self, tracker_len, metrics_kwargs):
        current_rewards = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        current_lengths = torch.zeros(self.num_actors, dtype=int, device=self.device)
        current_scores = {'rewards': current_rewards, 'lengths': current_lengths}
        metrics = Metrics(current_scores, tracker_len, **metrics_kwargs, env_render=self.env_render)
        return metrics

    @contextmanager
    def _as_metrics(self, new_metrics):
        r"""Temporarily swaps Agent.metrics to new_metrics, useful for evaluation."""
        old_metrics = self.metrics
        self.metrics = new_metrics
        try:
            yield
        finally:
            self.metrics = old_metrics

    def _checkpoint_save(self, stat, stat_name='rewards', higher_better=True):
        if self.ckpt_every > 0 and (self.epoch + 1) % self.ckpt_every == 0:
            ckpt_name = f'epoch={self.epoch}_steps={self.agent_steps}_{stat_name}={stat:.2f}'
            self.save(os.path.join(self.ckpt_dir, ckpt_name + '.pth'))
            latest_ckpt_path = os.path.join(self.ckpt_dir, 'latest.pth')
            if os.path.exists(latest_ckpt_path):
                os.unlink(latest_ckpt_path)
            os.symlink(ckpt_name + '.pth', latest_ckpt_path)

        better = (stat > self.best_stat if higher_better else stat < self.best_stat) if self.best_stat is not None else True
        if better:
            print(f'saving current best_{stat_name}={stat:.2f}')
            if self.best_stat is not None:
                # remove previous best file
                prev_best_ckpt = os.path.join(self.ckpt_dir, f'best_{stat_name}={self.best_stat:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
            self.best_stat = stat
            self.save(os.path.join(self.ckpt_dir, f'best_{stat_name}={self.best_stat:.2f}.pth'))

    def _convert_obs(self, obs):
        if not isinstance(obs, dict):
            obs = {'obs': obs}

        # NOTE: copying obs dict since env.step may modify it (ie. IsaacGymEnvs)
        _obs = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                _obs[k] = torch.tensor(v, device=self.device if not re.match(self.cpu_obs_keys, k) else 'cpu')
            else:
                # assert isinstance(v, torch.Tensor)
                _obs[k] = v
        return _obs

    @staticmethod
    def _handle_timeout(dones, info, timeout_keys=('time_outs', 'TimeLimit.truncated')):
        timeout_envs = None
        for timeout_key in timeout_keys:
            if timeout_key in info:
                timeout_envs = info[timeout_key]
                break
        if timeout_envs is not None:
            dones = dones * (~timeout_envs)
        return dones
