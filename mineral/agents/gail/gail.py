import json
import os
import re
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ppo.ppo import PPO
from .models import Discriminator


class GAIL(PPO):
    """Generative Adversarial Imitation Learning built on PPO with minimal changes."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)

        # GAIL-specific config
        self.gail_config = full_cfg.agent.get("gail", {})

        # demos
        demos_path = self.gail_config.get("demos_path", "")
        assert isinstance(demos_path, str) and len(demos_path) > 0, "GAIL requires agent.gail.demos_path"
        assert os.path.exists(demos_path), f"GAIL demos path: {demos_path} does not exist"
        demos = torch.load(demos_path, map_location=self.device)

        # support either a packed tensor dict (obs, act) or dict of episodes
        if isinstance(demos, dict) and "obs" in demos and "act" in demos:
            expert_obs = demos["obs"]  # [N, T, obs_dim] or [B, obs_dim]
            expert_act = demos["act"]  # [N, T, act_dim] or [B, act_dim]
            if expert_obs.dim() == 3:
                N, T, _ = expert_obs.shape
                expert_obs = expert_obs.reshape(N * T, -1)
                expert_act = expert_act.reshape(N * T, -1)
        elif isinstance(demos, dict) and all(isinstance(k, (int, str)) for k in demos.keys()):
            # dict of episodes {idx: {obs: [T, obs_dim], act: [T, act_dim], ...}}
            obs_list, act_list = [], []
            for ep in demos.values():
                if "obs" in ep and "act" in ep:
                    obs_list.append(ep["obs"])  # [T, obs_dim]
                    act_list.append(ep["act"])  # [T, act_dim]
            assert len(obs_list) > 0, "No (obs, act) pairs found in demos"
            expert_obs = torch.cat(obs_list, dim=0)
            expert_act = torch.cat(act_list, dim=0)
        else:
            raise ValueError("Unsupported demos format for GAIL: expected keys ('obs','act') or dict of episodes")

        # ensure tensors on device
        self.expert_obs = expert_obs.to(self.device).float()
        self.expert_act = expert_act.to(self.device).float()

        # discriminator
        obs_dim = self.obs_space["obs"]
        obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
        act_dim = self.action_dim
        self.discriminator = Discriminator(obs_dim, act_dim, self.gail_config.get("discriminator_mlp", None)).to(
            self.device
        )

        disc_optim_kwargs = self.gail_config.get("discriminator_optim", {"type": "Adam", "kwargs": {"lr": 3e-4}})
        DiscOptim = getattr(torch.optim, disc_optim_kwargs.get("type", "Adam"))
        self.disc_optim = DiscOptim(self.discriminator.parameters(), **disc_optim_kwargs.get("kwargs", {}))

        # training params
        self.disc_iters = int(self.gail_config.get("discriminator_iters", 1))
        self.expert_batch_size = int(self.gail_config.get("expert_batch_size", self.minibatch_size))
        self.policy_batch_size = int(self.gail_config.get("policy_batch_size", self.minibatch_size))
        self.reward_scale = float(self.gail_config.get("reward_scale", 1.0))
        self.label_smooth = float(self.gail_config.get("label_smooth", 0.0))

    @torch.no_grad()
    def gail_reward(self, obs, actions):
        # obs: dict of tensors with key 'obs' -> [B, obs_dim]
        logits = self.discriminator(obs, actions)  # [B]
        probs = torch.sigmoid(logits)
        reward = -torch.log(1.0 - probs + 1e-6)
        return (reward * self.reward_scale).unsqueeze(-1)

    @torch.no_grad()
    def play_steps(self):
        # identical to PPO.play_steps except rewards come from discriminator
        for n in range(self.horizon_len):
            if not self.env_autoresets:
                if any(self.dones):
                    done_indices = torch.where(self.dones)[0].tolist()
                    obs_reset = self.env.reset_idx(done_indices)
                    obs_reset = self._convert_obs(obs_reset)
                    for k, v in obs_reset.items():
                        self.obs[k][done_indices] = v

            model_out = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs)
            for k in ['actions', 'neglogp', 'values', 'mu', 'sigma']:
                self.storage.update_data(k, n, model_out[k])

            # do env step
            actions = torch.clamp(model_out['actions'], -1.0, 1.0)
            obs, r, self.dones, infos = self.env.step(actions)
            self.obs = self._convert_obs(obs)
            r, self.dones = torch.tensor(r, device=self.device), torch.tensor(self.dones, device=self.device)

            # GAIL reward from discriminator using pre-step obs and taken actions
            shaped_rewards = self.gail_reward(self.storage.storage_dict['obses']['obs'][n], model_out['actions'])

            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            if self.value_bootstrap and 'time_outs' in infos:
                time_outs = torch.tensor(infos['time_outs'], device=self.device)
                time_outs = time_outs.reshape(-1, 1)
                shaped_rewards += self.gamma * model_out['values'] * time_outs.float()
            self.storage.update_data('rewards', n, shaped_rewards)

            # still track environment reward for metrics
            rewards = r.reshape(-1, 1)
            done_indices = torch.where(self.dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards.squeeze(-1), done_indices, infos)
        self.metrics.flush_video(self.epoch)

        model_out = self.model_act(self.obs)
        last_values = model_out['values']

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

    def _sample_expert_batch(self, batch_size):
        N = self.expert_obs.shape[0]
        idx = torch.randint(0, N, (batch_size,), device=self.device)
        return self.expert_obs[idx], self.expert_act[idx]

    def train_discriminator(self):
        # use the flattened storage prepared in prepare_training
        data = self.storage.data_dict
        obs = data['obses']['obs']  # [B, obs_dim]
        act = data['actions']       # [B, act_dim]

        self.discriminator.train()
        losses = []
        for _ in range(self.disc_iters):
            # sample policy batch
            idx_pol = torch.randint(0, obs.size(0), (self.policy_batch_size,), device=self.device)
            pol_obs, pol_act = obs[idx_pol], act[idx_pol]
            # sample expert batch
            exp_obs, exp_act = self._sample_expert_batch(self.expert_batch_size)

            # forward
            pol_logits = self.discriminator(pol_obs, pol_act)
            exp_logits = self.discriminator(exp_obs, exp_act)

            # labels with optional smoothing
            real_label = 1.0 - self.label_smooth
            fake_label = 0.0 + self.label_smooth
            loss_real = F.binary_cross_entropy_with_logits(exp_logits, torch.full_like(exp_logits, real_label))
            loss_fake = F.binary_cross_entropy_with_logits(pol_logits, torch.full_like(pol_logits, fake_label))
            loss = loss_real + loss_fake

            self.disc_optim.zero_grad()
            loss.backward()
            self.disc_optim.step()
            losses.append(loss.detach())

        self.discriminator.eval()
        return {"disc_loss": torch.stack(losses).mean() if len(losses) > 0 else torch.tensor(0.0)}

    def train(self):
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1

            self.set_eval()
            self.play_steps()
            self.agent_steps += self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size

            # discriminator update step(s)
            disc_metrics = self.train_discriminator()

            self.set_train()
            results = self.train_epoch()  # reuse PPO training on shaped rewards
            self.storage.data_dict = None

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                # train metrics
                metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
                metrics.update({k: torch.mean(torch.cat(results[k]), 0).cpu().numpy() for k in ['mu', 'sigma']})
                metrics.update(
                    {
                        'epoch': self.epoch,
                        'mini_epoch': self.mini_epoch,
                        'last_lr': self.last_lr,
                        'e_clip': self.e_clip,
                        'disc_loss': disc_metrics["disc_loss"].item(),
                    }
                )
                metrics = {f'train_stats/{k}': v for k, v in metrics.items()}

                # timing
                timings = self.timer.stats(step=self.agent_steps, total_names=self.timer_total_names, reset=False)
                timing_metrics = {f'train_timings/{k}': v for k, v in timings.items()}
                metrics.update(timing_metrics)

                # episode metrics
                episode_metrics = {
                    'train_scores/episode_rewards': self.metrics.episode_trackers['rewards'].mean(),
                    'train_scores/episode_lengths': self.metrics.episode_trackers['lengths'].mean(),
                    'train_scores/num_episodes': self.metrics.num_episodes,
                    **self.metrics.result(prefix='train'),
                }
                metrics.update(episode_metrics)

                self.writer.add(self.agent_steps, metrics)
                self.writer.write()

                self._checkpoint_save(metrics['train_scores/episode_rewards'])

