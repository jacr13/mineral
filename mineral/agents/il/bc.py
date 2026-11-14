"""BC: behavioural cloning of the expert's (s, a) pairs (uses the expert actions: not state-only; reference).

Deterministic tanh-MLP policy, inputs standardised with the demo statistics, MSE regression; evaluated every round
on the env (deterministic, episodes_per_env episodes per env, as SHAC/FOCUS/MENTOR). Subclass: BCO.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nets import MLP
from ..agent import Agent
from .common import ExpertTransitions


class ClonedPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, mean, std, units=(256, 256)):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.net = MLP(obs_dim, out_dim=act_dim, units=list(units), plain_last=True, act_type='ELU')

    def forward(self, obs):
        return torch.tanh(self.net((obs - self.mean) / self.std))


class BC(Agent):
    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.bc_config = cfg = full_cfg.agent.bc
        self.num_actors = int(cfg.num_actors)
        super().__init__(full_cfg, **kwargs)
        assert 'obs' in self.obs_space, f'{type(self).__name__} supports flat observations only'
        self.expert = ExpertTransitions(self.device, cfg.demos)
        self.expert_return = self.expert.expert_return
        s = self.expert.obs['obs'].float()
        self.policy = ClonedPolicy(
            s.shape[-1], self.action_dim, s.mean(0), s.std(0).clamp_min(1e-3), self.network_config.get('units', [256, 256])
        ).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=float(cfg.lr))
        self.t0 = time.time()

    def fit(self, obs, actions, steps):
        self.policy.train()
        n, losses = obs.shape[0], []
        bs = int(self.bc_config.batch_size)
        for _ in range(int(steps)):
            i = torch.randint(0, n, (bs,), device=self.device)
            loss = F.mse_loss(self.policy(obs[i]), actions[i])
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.detach())
        self.policy.eval()
        return torch.stack(losses).mean().item()

    def train(self):
        cfg = self.bc_config
        for self.epoch in range(int(cfg.rounds)):
            loss = self.fit(self.expert.obs['obs'], self.expert.act, cfg.steps_per_round)
            self._log_round({'train/bc_loss': loss})
        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

    # ---- evaluation / logging ----
    @torch.no_grad()
    def evaluate(self, episodes_per_env=2):
        env = self.env
        obs = self._convert_obs(env.reset())['obs']
        n = self.num_actors
        ret = torch.zeros(n, device=self.device)
        length = torch.zeros(n, dtype=torch.long, device=self.device)
        count = torch.zeros(n, dtype=torch.long, device=self.device)
        returns, lengths = [], []
        T = int(getattr(env, 'episode_length', getattr(env, 'max_episode_length', 1000)))
        for _ in range(episodes_per_env * T + 10):
            o, r, d, _ = env.step(self.policy(obs))
            obs = self._convert_obs(o)['obs'].clone()
            ret += r
            length += 1
            for i in torch.as_tensor(d).nonzero(as_tuple=False).flatten().tolist():
                if count[i] < episodes_per_env:
                    returns.append(ret[i].item())
                    lengths.append(int(length[i].item()))
                count[i] += 1
                ret[i] = 0.0
                length[i] = 0
            if bool((count >= episodes_per_env).all()):
                break
        return returns, lengths

    def _log_round(self, stats):
        returns, lengths = self.evaluate(int(self.bc_config.get('eval_episodes_per_env', 2)))
        mean_ret = float(np.mean(returns))
        metrics = {
            'round': self.epoch,
            **stats,
            'time/wall': time.time() - self.t0,
            'eval_scores/num_episodes': len(returns),
            'eval_scores/episode_rewards': mean_ret,
            'eval_scores/episode_lengths': float(np.mean(lengths)),
            'eval_scores/normalized_return': mean_ret / self.expert_return,
        }
        self.writer.add(self.agent_steps, metrics)
        self.writer.write()
        print(f'[{type(self).__name__}] round {self.epoch}: return {mean_ret:.0f} ({mean_ret / self.expert_return:.3f})')
        scores = {
            'epoch': self.epoch,
            'agent_steps': self.agent_steps,
            'expert_return': self.expert_return,
            'eval_scores/num_episodes': len(returns),
            'eval_scores/episode_rewards': returns,
            'eval_scores/episode_lengths': lengths,
        }
        json.dump(scores, open(os.path.join(self.logdir, 'scores.json'), 'w'), indent=4)
        self._checkpoint_save(mean_ret)

    def eval(self):
        self._log_round({})

    def set_train(self):
        self.policy.train()

    def set_eval(self):
        self.policy.eval()

    def save(self, f):
        torch.save({'policy': self.policy.state_dict(), 'epoch': self.epoch, 'agent_steps': self.agent_steps}, f)

    def load(self, f, ckpt_keys=''):
        ckpt = torch.load(f, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
