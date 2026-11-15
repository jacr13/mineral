"""GAIfO: Generative Adversarial Imitation from Observation (Torabi et al., 2018), on PPO.

The discriminator D(s, s') separates expert transitions from the agent's; the agent maximises r = -log(1 - D(s, s'))
with PPO. Per rollout: collect H steps (storing the true successor of each step, also at resets), train the
discriminator on that rollout, then score every transition with the updated discriminator and run PPO.
Subclasses: GAIL (D(s, a), uses expert actions), MAAD (adds an inverse-dynamics regulariser).
"""

import torch

from ..ppo.ppo import PPO
from .common import ExpertTransitions, adversarial_reward, discriminator_loss, slice_obs, sliced_space, state_dims, to_device
from .models import TransitionDiscriminator, gradient_penalty


class GAIfO(PPO):
    USE_ACTIONS = False  # GAIL: D(s, a)

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        self.il_config = il = full_cfg.agent.il
        self.expert = ExpertTransitions(self.device, il.demos)

        self.state_d = state_dims(full_cfg, self.obs_space, il.get('state_obs', False))
        dcfg = il.discriminator
        self.discriminator = TransitionDiscriminator(
            sliced_space(self.obs_space, self.state_d),
            self.action_dim,
            use_actions=self.USE_ACTIONS,
            mlp_kwargs=dcfg.get('mlp_kwargs', None),
            encoder_kwargs=dcfg.get('encoder_kwargs', None) or full_cfg.agent.network.get('encoder_kwargs', None),
        ).to(self.device)
        self.disc_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=float(dcfg.get('lr', 3e-4)), weight_decay=float(dcfg.get('weight_decay', 0.0))
        )
        self.disc_iters = int(dcfg.get('iters', 10))
        self.disc_batch = int(dcfg.get('batch_size', 1024))
        self.disc_update_freq = int(dcfg.get('update_freq', 1))
        self.label_smooth = float(dcfg.get('label_smooth', 0.0))
        self.grad_pen = float(dcfg.get('grad_penalty', 0.0))
        self.reward_kind = il.get('reward', 'gail')
        self.reward_scale = float(il.get('reward_scale', 1.0))

        # true successor s' of every rollout step (the env's pre-reset observation at done steps)
        self.next_obs_buf = {k: torch.zeros_like(v, device=self.device) for k, v in self.storage.storage_dict['obses'].items()}
        self.disc_metrics = {}

    # ---- helpers ----
    def _norm_full(self, obs):
        """Policy input: the whole observation, normalised."""
        obs = to_device(obs, self.device)
        if self.normalize_input:
            return {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        return obs

    def _norm(self, obs):
        """Discriminator / inverse-model input: the state part of the normalised observation."""
        return slice_obs(self._norm_full(obs), self.state_d)

    def _second(self, actions, next_obs):
        return actions.clamp(-1, 1) if self.USE_ACTIONS else self._norm(next_obs)

    def _policy_pairs(self, idx):
        """Agent transitions (s, a, true s') at flat indices of the (horizon, num_envs) rollout."""
        n, e = idx // self.num_actors, idx % self.num_actors
        obs = {k: v[n.to(v.device), e.to(v.device)] for k, v in self.storage.storage_dict['obses'].items()}
        next_obs = {k: v[n, e] for k, v in self.next_obs_buf.items()}
        return obs, self.storage.storage_dict['actions'][n, e], next_obs

    def sample_valid(self, n):
        return self.valid_idx[torch.randint(0, len(self.valid_idx), (n,), device=self.device)]

    # ---- discriminator ----
    @torch.enable_grad()
    def train_discriminator(self):
        self.discriminator.train()
        logs = []
        for _ in range(self.disc_iters):
            p_obs, p_act, p_next = self._policy_pairs(self.sample_valid(self.disc_batch))
            e_obs, e_act, e_next, _ = self.expert.sample(self.disc_batch)
            x_p = self.discriminator.features(self._norm(p_obs), self._second(p_act, p_next))
            x_e = self.discriminator.features(self._norm(e_obs), self._second(e_act, e_next))
            loss, m = discriminator_loss(
                self.discriminator.logits_from_features(x_e), self.discriminator.logits_from_features(x_p), self.label_smooth
            )
            if self.grad_pen > 0:
                gp = gradient_penalty(self.discriminator, x_e, x_p)
                loss = loss + self.grad_pen * gp
                m['grad_penalty'] = gp.detach()
            self.disc_optim.zero_grad()
            loss.backward()
            self.disc_optim.step()
            m['loss'] = loss.detach()
            logs.append(m)
        self.discriminator.eval()
        return {f'discriminator/{k}': torch.stack([m[k] for m in logs]).mean().item() for k in logs[0]}

    @torch.no_grad()
    def imitation_reward(self, obs, actions, next_obs):
        logits = self.discriminator(self._norm(obs), self._second(actions, next_obs))
        return self.reward_scale * adversarial_reward(logits, self.reward_kind)

    # ---- rollout ----
    @torch.no_grad()
    def play_steps(self):
        timeouts = torch.zeros(self.horizon_len, self.num_actors, device=self.device)
        finite = torch.ones(self.horizon_len, self.num_actors, dtype=torch.bool, device=self.device)
        boot_values = torch.zeros(self.horizon_len, self.num_actors, 1, device=self.device)
        for n in range(self.horizon_len):
            if not self.env_autoresets:
                if any(self.dones):
                    done_indices = torch.where(self.dones)[0].tolist()
                    obs_reset = self._convert_obs(self.env.reset_idx(done_indices))
                    for k, v in obs_reset.items():
                        self.obs[k][done_indices] = v

            model_out = self.model_act(self.obs)
            self.storage.update_data('obses', n, self.obs)
            for k in ['actions', 'neglogp', 'values', 'mu', 'sigma']:
                self.storage.update_data(k, n, model_out[k])
            actions = torch.clamp(model_out['actions'], -1.0, 1.0)
            obs, r, dones, infos = self.env.step(actions)
            next_obs = self._convert_obs(obs)
            dones = torch.as_tensor(dones, device=self.device)
            real_next = self._real_next_obs(next_obs, dones, infos)
            next_obs, real_next, dones, _ = self._reset_nonfinite(self.obs, next_obs, real_next, dones)
            for k, v in real_next.items():
                self.next_obs_buf[k][n] = v.to(self.device)
            for o in (self.obs, real_next):
                for v in o.values():
                    finite[n] &= torch.isfinite(v.reshape(v.shape[0], -1)).all(-1).to(self.device)
            self.obs, self.dones = next_obs, dones
            self.storage.update_data('dones', n, self.dones)
            if self.value_bootstrap and 'time_outs' in infos:
                to = torch.as_tensor(infos['time_outs'], device=self.device).float().view(-1)
                timeouts[n] = to
                if to.any():  # bootstrap a time-limit end with V(true s'), not V(reset s)
                    boot_values[n] = self.model_act(real_next)['values']

            rewards = torch.as_tensor(r, device=self.device).reshape(-1)
            done_indices = torch.where(self.dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards, done_indices, infos)
        self.metrics.flush_video(self.epoch)

        self.valid_idx = torch.nonzero(finite.flatten()).flatten()  # agent pairs used to train D / inverse models
        if len(self.valid_idx) == 0:
            self.valid_idx = torch.arange(finite.numel(), device=self.device)
        if self.epoch % self.disc_update_freq == 0 or not self.disc_metrics:
            self.disc_metrics = self.train_discriminator()

        # score the whole rollout with the updated discriminator
        H, E = self.horizon_len, self.num_actors
        flat = torch.arange(H * E, device=self.device)
        rew = torch.empty(H * E, device=self.device)
        for c in range(0, H * E, 8192):
            o, a, no = self._policy_pairs(flat[c : c + 8192])
            rew[c : c + 8192] = self.imitation_reward(o, a, no)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0).view(H, E, 1)
        self.disc_metrics['imitation/reward_mean'] = rew.mean().item()
        rew = rew + self.gamma * boot_values * timeouts.unsqueeze(-1)
        self.storage.storage_dict['rewards'][:] = rew

        last_values = self.model_act(self.obs)['values']
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()
        values = self.storage.data_dict['values']
        returns = self.storage.data_dict['returns']
        if self.normalize_value:
            self.value_rms.update(values)
            values = self.value_rms.normalize(values)
            self.value_rms.update(returns)
            returns = self.value_rms.normalize(returns)
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def _extra_metrics(self):
        return {**self.disc_metrics, 'train_stats/expert_return': self.expert.expert_return}

    def save(self, f):
        super().save(f)
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        ckpt['discriminator'] = self.discriminator.state_dict()
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=''):
        super().load(f, ckpt_keys)
        ckpt = torch.load(f, map_location=self.device, weights_only=False)
        if 'discriminator' in ckpt:
            try:
                self.discriminator.load_state_dict(ckpt['discriminator'])
            except RuntimeError as e:  # e.g. a checkpoint from before the state-only inputs; the policy is loaded
                print(f'Warning: discriminator not loaded ({str(e).splitlines()[0]})')
