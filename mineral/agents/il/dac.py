"""DAC: Discriminator-Actor-Critic (Kostrikov et al., 2019), state-only variant D(s, s'), on SAC.

- 1-step transitions in the replay (the discriminator scores (s, s') pairs, expert pairs are 1-step);
- the reward is relabelled with the current discriminator for every sampled batch (not frozen at collection);
- absorbing states: a terminated transition (s, s'_T) leads to an absorbing state, so
  target = r(s, s'_T) + gamma [r(s'_T, abs) + gamma V_abs],  V_abs = r(abs, abs) / (1 - gamma),
  and the discriminator also sees (s'_T, abs) and (abs, abs) pairs (time-limit ends are not terminations);
- gradient penalty on the discriminator.
Subclass: OPOLO.
"""

import collections

import torch

from ..ddpg.utils import soft_update
from ..sac.sac import SAC
from .common import (
    ExpertTransitions,
    adversarial_reward,
    discriminator_loss,
    finite_rows,
    slice_obs,
    sliced_space,
    state_dims,
    tree_cat,
    tree_index,
)
from .models import TransitionDiscriminator, gradient_penalty


class DAC(SAC):
    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        assert int(self.sac_config.nstep) == 1, 'DAC scores 1-step (s, s\') pairs: set agent.sac.nstep=1'
        assert self.sac_config.handle_timeout, 'DAC needs handle_timeout (a time limit is not an absorbing state)'
        self.il_config = il = full_cfg.agent.il
        self.expert = ExpertTransitions(self.device, il.demos)

        self.state_d = state_dims(full_cfg, self.obs_space, il.get('state_obs', False))
        dcfg = il.discriminator
        self.discriminator = TransitionDiscriminator(
            sliced_space(self.obs_space, self.state_d),
            self.action_dim,
            use_actions=False,
            absorbing=True,
            mlp_kwargs=dcfg.get('mlp_kwargs', None),
            encoder_kwargs=dcfg.get('encoder_kwargs', None) or self.network_config.get('encoder_kwargs', None),
        ).to(self.device)
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=float(dcfg.get('lr', 1e-3)))
        self.disc_iters = int(dcfg.get('iters', 1))
        self.disc_batch = int(dcfg.get('batch_size', 256))
        self.grad_pen = float(dcfg.get('grad_penalty', 10.0))
        self.label_smooth = float(dcfg.get('label_smooth', 0.0))
        self.reward_kind = il.get('reward', 'airl')
        self.reward_scale = float(il.get('reward_scale', 1.0))
        self.gamma = float(self.sac_config.gamma)
        self.disc_metrics = {}

    def _norm(self, obs):
        """Policy / critic input: the whole observation, normalised."""
        if self.normalize_input:
            return {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        return obs

    def _dnorm(self, obs):
        """Discriminator / inverse-model input: the state part of the normalised observation."""
        return slice_obs(self._norm(obs), self.state_d)

    def _pairs(self, obs, next_obs, done):
        """(s, s') pairs of a batch plus, for its terminated rows, (s'_T, abs) and (abs, abs)."""
        done = done.view(-1).bool()
        n = done.shape[0]
        zeros = torch.zeros(n, device=self.device)
        o1, o2, a1, a2 = obs, next_obs, zeros, zeros
        if done.any():
            term = tree_index(next_obs, done)
            k = term[next(iter(term))].shape[0]
            z, one = torch.zeros(k, device=self.device), torch.ones(k, device=self.device)
            o1 = tree_cat(obs, term, term)
            o2 = tree_cat(next_obs, term, term)
            a1 = torch.cat([zeros, z, one])
            a2 = torch.cat([zeros, one, one])
        return self._dnorm(o1), self._dnorm(o2), a1, a2

    def train_discriminator(self, memory):
        self.discriminator.train()
        logs = []
        for _ in range(self.disc_iters):
            p_obs, _, _, p_next, p_done = memory.sample_batch(self.disc_batch, device=self.device)
            ok = finite_rows(p_obs, p_next)
            p_obs, p_next, p_done = tree_index(p_obs, ok), tree_index(p_next, ok), p_done[ok]
            e_obs, _, e_next, e_term = self.expert.sample(self.disc_batch)
            x_p = self.discriminator.features(*self._pairs(p_obs, p_next, p_done))
            x_e = self.discriminator.features(*self._pairs(e_obs, e_next, e_term))
            loss, m = discriminator_loss(
                self.discriminator.logits_from_features(x_e), self.discriminator.logits_from_features(x_p), self.label_smooth
            )
            gp = gradient_penalty(self.discriminator, x_e, x_p)
            loss = loss + self.grad_pen * gp
            self.disc_optim.zero_grad()
            loss.backward()
            self.disc_optim.step()
            m['grad_penalty'] = gp.detach()
            m['loss'] = loss.detach()
            logs.append(m)
        self.discriminator.eval()
        return {f'discriminator/{k}': torch.stack([m[k] for m in logs]).mean().item() for k in logs[0]}

    @torch.no_grad()
    def relabel(self, obs, next_obs, done):
        """Imitation reward of a replay batch, absorbing-state value folded in for terminated rows."""
        d = self.discriminator
        o, no = self._dnorm(obs), self._dnorm(next_obs)
        r = adversarial_reward(d(o, no), self.reward_kind)
        n = r.shape[0]
        zeros, ones = torch.zeros(n, device=self.device), torch.ones(n, device=self.device)
        r_term = adversarial_reward(d(no, no, zeros, ones), self.reward_kind)  # (s'_T, abs)
        first = {k: v[:1] for k, v in no.items()}
        r_abs = adversarial_reward(d(first, first, ones[:1], ones[:1]), self.reward_kind)  # (abs, abs)
        v_abs = r_abs / (1 - self.gamma)
        done = done.view(-1).float()
        r = r + done * self.gamma * (r_term + self.gamma * v_abs)
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return (self.reward_scale * r).unsqueeze(-1), r_abs

    def update_net(self, memory):
        self.disc_metrics = self.train_discriminator(memory)
        results = collections.defaultdict(list)
        for _i in range(self.sac_config.mini_epochs):
            self.mini_epoch += 1
            obs, action, _, next_obs, done = memory.sample_batch(self.sac_config.batch_size, device=self.device)
            reward, r_abs = self.relabel(obs, next_obs, done)
            results['imitation/reward_mean'].append(reward.mean())
            results['imitation/reward_absorbing'].append(r_abs.mean())

            critic_loss, critic_grad_norm, target_values = self.update_critic(obs, action, reward, next_obs, done)
            results["loss/critic"].append(critic_loss)
            if critic_grad_norm is not None:
                results["grad_norm/critic"].append(critic_grad_norm)
            for k, v in target_values.items():
                results[k].append(v)

            if self.mini_epoch % self.sac_config.update_actor_interval == 0:
                actor_loss, alpha_loss, entropy, actor_grad_norm = self.update_actor(obs)
                results["loss/actor"].append(actor_loss)
                if alpha_loss is not None:
                    results["loss/alpha"].append(alpha_loss)
                results["entropy"].append(entropy)
                if actor_grad_norm is not None:
                    results["grad_norm/actor"].append(actor_grad_norm)

            if self.mini_epoch % self.sac_config.update_targets_interval == 0:
                soft_update(self.critic_target, self.critic, self.sac_config.tau)
                if not self.sac_config.no_tgt_actor:
                    soft_update(self.actor_target, self.actor, self.sac_config.tau)
        return results

    def _extra_metrics(self):
        return {**self.disc_metrics, 'train_stats/expert_return': self.expert.expert_return}

    def save(self, f):
        super().save(f)
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        ckpt['discriminator'] = self.discriminator.state_dict()
        torch.save(ckpt, f)
