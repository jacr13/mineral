"""BCO(alpha): Behavioral Cloning from Observation (Torabi et al., 2018), state-only.

1. pre-demonstration: run a random policy for pre_steps steps per env, fit an inverse model g(s, s') -> a;
2. label every expert pair (s, s') with g, behaviour-clone the labels;
3. repeat for `rounds`: run the current policy (+ noise) for alpha_steps steps per env, add the transitions to the
   inverse model's data, refit g, relabel, clone. alpha_steps = 0 is BCO(0).
"""

import os

import torch

from .bc import BC
from .common import InverseActionModel, state_dims


class BCO(BC):
    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        inv = self.bc_config.inverse
        # the inverse model reads the state part only (DFlex obs end with the previous action, i.e. a_t is in s')
        self.inv_d = state_dims(full_cfg, self.obs_space) if self.obs_keys == ['obs'] else None
        d = self.inv_d or self.expert_s.shape[-1]
        self.inverse = InverseActionModel({'obs': (d,)}, self.action_dim, self.device, dict(inv))
        self.data = {'obs': [], 'act': [], 'next': []}
        self.obs = None

    def _norm(self, x):
        x = (x - self.policy.mean) / self.policy.std
        return {'obs': x if self.inv_d is None else x[..., : self.inv_d]}

    @torch.no_grad()
    def collect(self, steps, random):
        if self.obs is None:
            self.obs = self.vec(self._convert_obs(self.env.reset()))
        noise = float(self.bc_config.get('explore_noise', 0.1))
        for _ in range(int(steps)):
            if random:
                a = torch.rand(self.num_actors, self.action_dim, device=self.device) * 2 - 1
            else:
                a = self.policy(self.obs) + noise * torch.randn(self.num_actors, self.action_dim, device=self.device)
                a = a.clamp(-1, 1)
            o, _, d, infos = self.env.step(a)
            nxt = self._convert_obs(o)
            real = self.vec(self._real_next_obs(nxt, torch.as_tensor(d), infos))
            self.data['obs'].append(self.obs.clone())
            self.data['act'].append(a)
            self.data['next'].append(real.clone())
            self.obs = self.vec(nxt).clone()
            self.agent_steps += self.num_actors

    def fit_inverse(self):
        obs, act, nxt = (torch.cat(self.data[k]) for k in ('obs', 'act', 'next'))
        ok = torch.isfinite(obs).all(-1) & torch.isfinite(nxt).all(-1)
        obs, act, nxt = obs[ok], act[ok], nxt[ok]
        bs, losses = int(self.bc_config.inverse.batch_size), []
        for _ in range(int(self.bc_config.inverse.steps)):
            i = torch.randint(0, obs.shape[0], (bs,), device=self.device)
            losses.append(self.inverse.train_step(self._norm(obs[i]), act[i], self._norm(nxt[i])))
        return torch.stack(losses).mean().item()

    @torch.no_grad()
    def label_expert(self):
        s, s1 = self.expert_s, self.expert_s1
        return torch.cat(
            [self.inverse(self._norm(s[c : c + 65536]), self._norm(s1[c : c + 65536])) for c in range(0, s.shape[0], 65536)]
        )

    def train(self):
        cfg = self.bc_config
        self.collect(cfg.pre_steps, random=True)
        for self.epoch in range(int(cfg.rounds)):
            if self.epoch > 0 and int(cfg.alpha_steps) > 0:
                self.collect(cfg.alpha_steps, random=False)
            if self.epoch == 0 or int(cfg.alpha_steps) > 0:
                inv_loss = self.fit_inverse()
                labels = self.label_expert()
            with torch.no_grad():
                label_err = (labels - self.expert.act).pow(2).mean().item()  # diagnostic only (never trained on)
            loss = self.fit(self.expert_s, labels, cfg.steps_per_round)
            self._log_round({'train/bc_loss': loss, 'train/inverse_loss': inv_loss, 'audit/label_mse_vs_expert': label_err})
            self.obs = None  # evaluation reset the env
        self.save(os.path.join(self.ckpt_dir, 'final.pth'))
