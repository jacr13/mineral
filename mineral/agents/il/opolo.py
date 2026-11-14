"""OPOLO: Off-Policy Learning from Observations (Zhu et al., NeurIPS 2020), on the state-only DAC.

Adds to DAC (official code: illidanlab/opolo-code, td3opolo.py):
- the AlgaeDICE initial-state term 2 alpha (1 - gamma) E[Q(s, pi(s))] in the critic loss (replay states as s_0);
- an inverse-action regulariser on the policy: reg_coef ||E[pi(s_E)] - g(s_E, s'_E)||^2, with the inverse model
  g fitted on the agent's replay transitions.
"""

import torch
import torch.nn.functional as F

from .common import InverseActionModel, finite_rows, sliced_space, tree_index
from .dac import DAC


class OPOLO(DAC):
    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        cfg = full_cfg.agent.opolo
        self.opolo_alpha = float(cfg.get('alpha', 0.1))
        inv = cfg.inverse
        enc = self.network_config.get('encoder_kwargs', None)
        space = sliced_space(self.obs_space, self.state_d)
        self.inverse = InverseActionModel(space, self.action_dim, self.device, {'encoder_kwargs': enc, **inv})
        self.inv_reg_coef = float(inv.reg_coef)
        self.inv_batch = int(inv.batch_size)
        self.inv_iters = int(inv.iters)

    def update_net(self, memory):
        losses = []
        for _ in range(self.inv_iters):
            obs, act, _, next_obs, _ = memory.sample_batch(self.inv_batch, device=self.device)
            ok = finite_rows(obs, next_obs)
            obs, act, next_obs = tree_index(obs, ok), act[ok], tree_index(next_obs, ok)
            losses.append(self.inverse.train_step(self._dnorm(obs), act, self._dnorm(next_obs)))
        results = super().update_net(memory)
        if losses:
            results['loss/inverse_model'].append(torch.stack(losses).mean())
        return results

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_z = self.encoder_target(self._norm(next_obs))
            next_actions, _, log_prob = self.get_actions(z=next_z, logprob=True)
            target_Q = self.critic_target.get_q_min(next_z, next_actions)
            if self.sac_config.backup_entropy:
                target_Q -= self.get_alpha() * log_prob
            target_Q = reward + (1 - done) * self.gamma * target_Q

        z = self.encoder(self._norm(obs))
        current_Qs = self.critic.get_q_values(z, action)
        critic_loss = torch.sum(torch.stack([F.mse_loss(q, target_Q) for q in current_Qs]))
        with torch.no_grad():
            pi_actions = self.get_actions(z=z, sample=True)
        qf_init = self.critic.get_q_min(z, pi_actions).mean()
        dice_term = 2.0 * self.opolo_alpha * (1.0 - self.gamma) * qf_init
        critic_loss = critic_loss + dice_term
        grad_norm = self.optimizer_update(self.critic_optim, critic_loss)
        target_values = {
            "target_values/mean": target_Q.mean(),
            "target_values/std": target_Q.std(),
            "target_values/max": target_Q.max(),
            "target_values/min": target_Q.min(),
            "opolo/qf_init": qf_init.detach(),
        }
        return critic_loss, grad_norm, target_values

    def update_actor(self, obs):
        self.critic.requires_grad_(False)
        z = self.encoder(self._norm(obs))
        actions, _, log_prob = self.get_actions(z=z, logprob=True)
        Q = self.critic.get_q_min(z, actions)
        actor_loss = (self.get_alpha() * log_prob - Q).mean()

        e_obs, _, e_next, _ = self.expert.sample(self.inv_batch)
        target = self.inverse(self._dnorm(e_obs), self._dnorm(e_next))
        e_actions = self.get_actions(z=self.encoder(self._norm(e_obs)), sample=False)  # squashed mean, in [-1, 1]
        inv_reg = F.mse_loss(e_actions, target)
        actor_loss = actor_loss + self.inv_reg_coef * inv_reg
        self.last_inv_reg = inv_reg.detach()

        grad_norm = self.optimizer_update(self.actor_optim, actor_loss)
        self.critic.requires_grad_(True)
        entropy = -log_prob
        alpha_loss = None
        if self.sac_config.alpha is None:
            alpha_loss = (self.get_alpha(detach=False) * (entropy - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, alpha_loss)
        return actor_loss, alpha_loss, entropy.mean(), grad_norm

    def _extra_metrics(self):
        m = super()._extra_metrics()
        if hasattr(self, 'last_inv_reg'):
            m['opolo/inv_reg'] = self.last_inv_reg.item()
        return m
