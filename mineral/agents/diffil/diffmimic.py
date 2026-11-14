"""DiffMimic (Ren et al., ICLR 2023): state matching with analytic gradients through the simulator, DFlex.

Every iteration, each of ``batch`` simulator copies is reset to a demo state s^i_t (random demo and phase), the
deterministic policy is rolled out ``horizon`` steps through the differentiable simulator, and the loss is the
time-aligned state-tracking error sum_k ||W (x_k - s^i_{t+k})||^2 (W = 1 / per-coordinate demo spread).
Demonstration replay: when a copy's tracking error exceeds ``replay_threshold`` its simulator state is replaced by
the demo state (gradient cut there), so the rest of the window keeps learning from on-track states.
State-only: only demo states are used. Evaluated like MENTOR (deterministic policy, early termination, stochastic init).
"""

import json
import os
import time

import numpy as np
import torch

from ...common.demos import get_demos
from ..agent import Agent
from .mentor import MentorPolicy
from .mentor_sim import OneStep, SimCounter, get_spec


class DiffMimic(Agent):
    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.dm_config = cfg = full_cfg.agent.diffmimic
        self.num_actors = int(cfg.num_actors)
        super().__init__(full_cfg, **kwargs)
        self.seed = int(full_cfg.seed)
        self.spec = get_spec(full_cfg.task.env.env_name)
        D = self.spec.obs_dim

        demos = get_demos(self.device, **cfg.demos)
        obs = demos['obs']['obs'][..., :D].float()
        nxt = demos['next_obs']['obs'][..., :D].float()
        done = demos['done'].bool()
        N, T = done.shape
        first_done = torch.where(done.any(1), done.float().argmax(1), torch.full((N,), T - 1, device=self.device))
        finite = torch.isfinite(obs).all(-1) & torch.isfinite(nxt).all(-1)
        bad = (~finite).float().cumsum(1) > 0  # cut a demo before its first non-finite transition
        first_bad = torch.where(bad.any(1), bad.float().argmax(1) - 1, torch.full((N,), T - 1, device=self.device))
        self.lengths = torch.minimum(first_done, first_bad) + 1  # usable transitions per demo
        self.S = torch.cat([obs[:, :1], nxt], 1)  # states s_0 .. s_T, s_{t+1} = next_obs_t
        self.expert_return = demos['expert_return']
        valid = torch.arange(T, device=self.device)[None] < self.lengths[:, None]
        flat = obs[valid]
        self.w = 1.0 / flat.std(0).clamp_min(1e-3)

        self.policy = MentorPolicy(
            D, self.spec.act_dim, flat.mean(0), flat.std(0).clamp_min(1e-3), units=tuple(self.network_config.units)
        ).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=float(cfg.lr))
        self.counter = SimCounter()
        self.batch = int(cfg.batch)
        self.horizon = int(cfg.horizon)
        self.step = OneStep(self.spec, self.batch, self.device, self.counter)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)
        self.t0 = time.time()

    def _sample_starts(self):
        L = self.lengths
        ok = torch.nonzero(L > self.horizon).flatten()
        assert len(ok) > 0, f'no demo longer than horizon {self.horizon}'
        n = ok[torch.randint(0, len(ok), (self.batch,), device=self.device, generator=self.generator)]
        u = torch.rand(self.batch, device=self.device, generator=self.generator)
        t = (u * (L[n] - self.horizon + 1).float()).long()
        return n, t

    def update(self):
        n, t = self._sample_starts()
        x = self.S[n, t]
        loss, replays, err_sum = 0.0, 0.0, 0.0
        thr = float(self.dm_config.replay_threshold)
        for k in range(self.horizon):
            x = self.step(x, self.policy(x))
            ref = self.S[n, t + k + 1]
            err = (((x - ref) * self.w) ** 2).mean(-1)  # per copy, normalised per coordinate
            loss = loss + torch.nan_to_num(err, nan=0.0, posinf=0.0).mean()
            err_sum += err.detach().mean().item()
            if thr > 0:
                off = (err.detach() > thr) | ~torch.isfinite(err.detach())
                if off.any():
                    x = torch.where(off[:, None], ref, x)  # demonstration replay: back onto the demo
                    replays += off.float().mean().item()
        loss = loss / self.horizon
        self.optim.zero_grad()
        loss.backward()  # always: the simulator tape must be consumed
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.dm_config.max_grad_norm))
        if torch.isfinite(gn):
            self.optim.step()
        self.agent_steps += self.batch * self.horizon
        return {
            'train/loss': float(loss.detach()),
            'train/replay_rate': replays / self.horizon,
            'train/tracking_error': err_sum / self.horizon,
            'train/grad_norm': float(gn),
        }

    def train(self):
        cfg = self.dm_config
        stats = []
        for self.epoch in range(int(cfg.iters)):
            if self.job_clock is not None:
                _, safe_stop = self.job_clock.step(check_safe_stop=True)
                if safe_stop:
                    break
            stats.append(self.update())
            if (self.epoch + 1) % int(cfg.eval_every) == 0:
                self._log({k: float(np.mean([s[k] for s in stats])) for k in stats[0]})
                stats = []
        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

    @torch.no_grad()
    def evaluate(self, episodes_per_env=2):
        if not hasattr(self, 'eval_env'):
            env_cfg = self.full_cfg.task.env
            self.eval_env = self.spec.make_env(
                self.num_actors,
                self.device,
                seed=self.seed + 12345,
                stochastic_init=bool(env_cfg.get('stochastic_init', True)),
                early_termination=bool(env_cfg.get('early_termination', True)),
            )
        env, D, n = self.eval_env, self.spec.obs_dim, self.num_actors
        self.policy.eval()
        obs = self._convert_obs(env.reset())['obs'][:, :D]
        ret = torch.zeros(n, device=self.device)
        length = torch.zeros(n, dtype=torch.long, device=self.device)
        count = torch.zeros(n, dtype=torch.long, device=self.device)
        returns, lengths = [], []
        for _ in range(episodes_per_env * env.episode_length + 10):
            o, r, d, _ = env.step(self.policy(obs))
            obs = self._convert_obs(o)['obs'][:, :D].clone()
            ret += r
            length += 1
            for i in d.nonzero(as_tuple=False).flatten().tolist():
                if count[i] < episodes_per_env:
                    returns.append(ret[i].item())
                    lengths.append(int(length[i].item()))
                count[i] += 1
                ret[i] = 0.0
                length[i] = 0
            if bool((count >= episodes_per_env).all()):
                break
        self.policy.train()
        return returns, lengths

    def _log(self, stats):
        returns, lengths = self.evaluate(int(self.dm_config.get('eval_episodes_per_env', 2)))
        mean_ret = float(np.mean(returns))
        metrics = {
            'iter': self.epoch + 1,
            **stats,
            'time/wall': time.time() - self.t0,
            'sim_steps/total': int(self.counter.total),
            'eval_scores/num_episodes': len(returns),
            'eval_scores/episode_rewards': mean_ret,
            'eval_scores/episode_lengths': float(np.mean(lengths)),
            'eval_scores/normalized_return': mean_ret / self.expert_return,
        }
        self.writer.add(self.agent_steps, metrics)
        self.writer.write()
        print(
            f'[DiffMimic] iter {self.epoch + 1}: return {mean_ret:.0f} ({mean_ret / self.expert_return:.3f}), '
            f'loss {stats.get("train/loss", float("nan")):.3g}, replay {stats.get("train/replay_rate", 0):.2f}'
        )
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
        self._log({})

    def set_train(self):
        self.policy.train()

    def set_eval(self):
        self.policy.eval()

    def save(self, f):
        torch.save({'policy': self.policy.state_dict(), 'epoch': self.epoch, 'agent_steps': self.agent_steps}, f)

    def load(self, f, ckpt_keys=''):
        self.policy.load_state_dict(torch.load(f, map_location=self.device)['policy'])
