import os

import torch
import torch.nn.functional as F

from ..ppo.ppo import PPO
from .models import Discriminator


class GAIL(PPO):
    """Generative Adversarial Imitation Learning built on PPO with minimal changes."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)

        # GAIL-specific config
        self.gail_config = full_cfg.agent.get("gail", {})

        self.input_type = self.gail_config.input_type

        # demos
        demos_path = self.gail_config.demos.path
        assert os.path.exists(demos_path), f"Demos path: {demos_path} does not exist"
        self.demos = torch.load(demos_path, map_location=self.device)

        n_envs = self.gail_config.demos.num
        self.demos["obs"] = {k: v[:n_envs, ...] for k, v in self.demos["obs"].items()}
        self.demos["act"] = self.demos["act"][:n_envs, ...]
        self.demos["rew"] = self.demos["rew"][:n_envs, ...]
        self.expert_return = self.demos["rew"].sum(dim=1).mean().item()

        # discriminator
        discriminator_config = self.gail_config.get("discriminator", {})
        act_dim = self.action_dim
        self.discriminator = Discriminator(
            self.obs_space,
            act_dim,
            input_type=self.input_type,
            discriminator_kwargs=discriminator_config,
        ).to(self.device)

        disc_optim_kwargs = discriminator_config.get("optim", {"type": "Adam", "kwargs": {"lr": 3e-4}})
        DiscOptim = getattr(torch.optim, disc_optim_kwargs.get("type", "Adam"))
        self.disc_optim = DiscOptim(self.discriminator.parameters(), **disc_optim_kwargs.get("kwargs", {}))

        # training params
        self.disc_iters = int(discriminator_config.get("iters", 1))
        self.expert_batch_size = int(self.gail_config.get("expert_batch_size", self.minibatch_size))
        self.policy_batch_size = int(self.gail_config.get("policy_batch_size", self.minibatch_size))
        self.reward_scale = float(self.gail_config.get("reward_scale", 1.0))
        self.label_smooth = float(self.gail_config.get("label_smooth", 0.0))

    def get_discriminator_inputs(self, obs, actions, next_obs):
        if self.input_type == "state":
            input_1, input_2 = obs, None
        elif self.input_type == "state_action":
            input_1, input_2 = obs, actions
        elif self.input_type == "state_state":
            input_1, input_2 = obs, next_obs
        else:
            raise NotImplementedError

        return input_1, input_2

    @torch.no_grad()
    def gail_reward(self, obs, actions, next_obs):
        input_1, input_2 = self.get_discriminator_inputs(obs, actions, next_obs)
        logits = self.discriminator(input_1, input_2)
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

            # GAIL reward from discriminator using pre-step obs and taken actions
            shaped_rewards = self.gail_reward(
                obs={k: v[n] for k, v in self.storage.storage_dict['obses'].items()},
                actions=model_out['actions'],
                next_obs=self.obs,
            )

            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            if self.value_bootstrap and 'time_outs' in infos:
                time_outs = infos['time_outs']
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
        num_trajs, length, _ = self.demos["act"].shape
        # Sample random environment and step indices
        trajs_idx = torch.randint(0, num_trajs, (batch_size,), device=self.device)
        steps_idx = torch.randint(0, length - 1, (batch_size,), device=self.device)

        expert_obs = {k: v[trajs_idx, steps_idx, ...] for k, v in self.demos["obs"].items()}
        expert_act = self.demos["act"][trajs_idx, steps_idx, ...]
        expert_next_obs = {k: v[trajs_idx, steps_idx + 1, ...] for k, v in self.demos["obs"].items()}
        return expert_obs, expert_act, expert_next_obs

    def _sample_batch(self, obs_rollout, actions, batch_size):
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(0)
            obs_rollout = {k: v.unsqueeze(0) for k, v in obs_rollout.items()}
        num_trajs, length, _ = actions.shape

        if length < 2:
            raise ValueError("Rollout length must be >= 2 to sample next_obs")

        # Sample random environment and step indices
        trajs_idx = torch.randint(0, num_trajs, (batch_size,), device=self.device)
        steps_idx = torch.randint(0, length - 1, (batch_size,), device=self.device)

        obs = {k: v[trajs_idx, steps_idx, ...].detach() for k, v in obs_rollout.items()}
        act = actions[trajs_idx, steps_idx, :].detach()
        next_obs = {k: v[trajs_idx, steps_idx + 1, ...].detach() for k, v in obs_rollout.items()}

        return obs, act, next_obs

    def train_discriminator(self):
        # use the flattened storage prepared in prepare_training
        data = self.storage.data_dict
        obs = data['obses']
        act = data['actions']

        print('Training discriminator...')
        self.discriminator.train()
        losses = []
        for _ in range(self.disc_iters):
            # sample policy batch
            pol_obs, pol_act, pol_next_obs = self._sample_batch(obs, act, self.policy_batch_size)

            # sample expert batch
            exp_obs, exp_act, exp_next_obs = self._sample_batch(self.demos["obs"], self.demos["act"], self.expert_batch_size)

            # forward
            print("policy")
            pol_input_1, pol_input_2 = self.get_discriminator_inputs(pol_obs, pol_act, pol_next_obs)
            pol_logits = self.discriminator(pol_input_1, pol_input_2)

            exp_input_1, exp_input_2 = self.get_discriminator_inputs(exp_obs, exp_act, exp_next_obs)
            exp_logits = self.discriminator(exp_input_1, exp_input_2)

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
