"""MAAD: GAIfO plus an inverse-dynamics regulariser (on-policy, PPO).

An inverse model g(s, s') -> a is fitted on the agent's own transitions; on expert pairs it proposes actions,
and the policy mean is pulled towards them: loss += reg_coef * ||mu(s_E) - g(s_E, s'_E)||^2.
"""

import torch
import torch.nn.functional as F

from .common import InverseActionModel, sliced_space
from .gaifo import GAIfO


class MAAD(GAIfO):
    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        cfg = full_cfg.agent.maad
        enc = full_cfg.agent.network.get('encoder_kwargs', None)
        space = sliced_space(self.obs_space, self.state_d)
        self.inverse = InverseActionModel(space, self.action_dim, self.device, {'encoder_kwargs': enc, **cfg})
        self.inv_reg_coef = float(cfg.reg_coef)
        self.inv_batch = int(cfg.batch_size)
        self.inv_iters = int(cfg.iters)
        self.inv_update_freq = int(cfg.get('update_freq', 1))

    def _before_mini_epoch(self, results):
        if self.mini_epoch % self.inv_update_freq != 0:
            return
        losses = []
        for _ in range(self.inv_iters):
            obs, act, next_obs = self._policy_pairs(self.sample_valid(self.inv_batch))
            losses.append(self.inverse.train_step(self._norm(obs), act, self._norm(next_obs)))
        results['loss/inverse_model'].append(torch.stack(losses).mean())

    def _aux_loss(self, results):
        e_obs, _, e_next, _ = self.expert.sample(self.inv_batch)
        target = self.inverse(self._norm(e_obs), self._norm(e_next))
        mu = self.model(self._norm_full(e_obs))['mu']
        reg = F.mse_loss(mu, target)
        results['loss/inv_reg'].append(reg.detach())
        return self.inv_reg_coef * reg

    def save(self, f):
        super().save(f)
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        ckpt['inverse'] = self.inverse.state_dict()
        torch.save(ckpt, f)
