import os

import torch
import torch.nn.functional as F

from ..gail.models import Discriminator
from ..sac.sac import SAC


class DAC(SAC):
    """Discriminator-Actor-Critic built on SAC with minimal changes."""

    def __init__(self, full_cfg, **kwargs):
        self.dac_config = full_cfg.agent.get("dac", {})
        super().__init__(full_cfg, **kwargs)

        # demos
        demos_path = self.dac_config.get("demos_path", "")
        assert isinstance(demos_path, str) and len(demos_path) > 0, "DAC requires agent.dac.demos_path"
        assert os.path.exists(demos_path), f"DAC demos path: {demos_path} does not exist"
        demos = torch.load(demos_path, map_location=self.device)

        # support either a packed tensor dict (obs, act) or dict of episodes
        if isinstance(demos, dict) and "obs" in demos and "act" in demos:
            expert_obs = demos["obs"]
            expert_act = demos["act"]
            if expert_obs.dim() == 3:
                num_eps, horizon, _ = expert_obs.shape
                expert_obs = expert_obs.reshape(num_eps * horizon, -1)
                expert_act = expert_act.reshape(num_eps * horizon, -1)
        elif isinstance(demos, dict) and all(isinstance(k, (int, str)) for k in demos.keys()):
            obs_list, act_list = [], []
            for episode in demos.values():
                if "obs" in episode and "act" in episode:
                    obs_list.append(episode["obs"])
                    act_list.append(episode["act"])
            assert len(obs_list) > 0, "No (obs, act) pairs found in demos"
            expert_obs = torch.cat(obs_list, dim=0)
            expert_act = torch.cat(act_list, dim=0)
        else:
            raise ValueError("Unsupported demos format for DAC: expected keys ('obs','act') or dict of episodes")

        self.expert_obs = expert_obs.to(self.device).float()
        self.expert_act = expert_act.to(self.device).float()

        obs_dim = self.obs_space["obs"]
        obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
        act_dim = self.action_dim
        self.discriminator = Discriminator(
            obs_dim, act_dim, self.dac_config.get("discriminator_mlp", None)
        ).to(self.device)

        disc_optim_cfg = self.dac_config.get("discriminator_optim", {"type": "Adam", "kwargs": {"lr": 3e-4}})
        DiscOptim = getattr(torch.optim, disc_optim_cfg.get("type", "Adam"))
        self.disc_optim = DiscOptim(self.discriminator.parameters(), **disc_optim_cfg.get("kwargs", {}))

        self.disc_iters = int(self.dac_config.get("discriminator_iters", 1))
        self.expert_batch_size = int(self.dac_config.get("expert_batch_size", self.sac_config.batch_size))
        self.policy_batch_size = int(self.dac_config.get("policy_batch_size", self.sac_config.batch_size))
        self.reward_scale = float(self.dac_config.get("reward_scale", 1.0))
        self.label_smooth = float(self.dac_config.get("label_smooth", 0.0))

    @torch.no_grad()
    def dac_reward(self, obs, actions):
        logits = self.discriminator(obs, actions)
        probs = torch.sigmoid(logits)
        reward = -torch.log(1.0 - probs + 1e-6)
        return (reward * self.reward_scale).unsqueeze(-1)

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        self.discriminator.eval()

        traj_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_actions = torch.empty((self.num_actors, timesteps) + (self.action_dim,), device=self.device)
        traj_rewards = torch.empty((self.num_actors, timesteps), device=self.device)
        traj_next_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_dones = torch.empty((self.num_actors, timesteps), device=self.device)

        for t in range(timesteps):
            if not self.env_autoresets:
                raise NotImplementedError

            if self.normalize_input:
                for key, value in self.obs.items():
                    self.obs_rms[key].update(value)

            current_obs = self.obs
            if random:
                actions = torch.rand((self.num_actors, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                actions = self.get_actions(obs=self.obs, sample=sample)

            shaped_rewards = self.dac_reward(current_obs, actions)

            next_obs, rewards, dones, infos = env.step(actions)
            next_obs = self._convert_obs(next_obs)

            done_indices = torch.where(dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards, done_indices, infos)

            if self.sac_config.handle_timeout:
                dones = self._handle_timeout(dones, infos)

            for key, value in self.obs.items():
                traj_obs[key][:, t] = value
            traj_actions[:, t] = actions
            traj_dones[:, t] = dones
            traj_rewards[:, t] = shaped_rewards.squeeze(-1)
            for key, value in next_obs.items():
                traj_next_obs[key][:, t] = value
            self.obs = next_obs

        self.metrics.flush_video(self.epoch)

        traj_rewards = self.reward_shaper(traj_rewards.reshape(self.num_actors, timesteps, 1))
        traj_dones = traj_dones.reshape(self.num_actors, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones)

        return data, timesteps * self.num_actors

    def _sample_expert_batch(self, batch_size):
        batch_size = min(batch_size, self.expert_obs.shape[0])
        indices = torch.randint(0, self.expert_obs.shape[0], (batch_size,), device=self.device)
        return self.expert_obs[indices], self.expert_act[indices]

    def _sample_policy_batch(self, batch_size):
        if self.memory.cur_capacity == 0:
            return None, None
        batch_size = min(batch_size, self.memory.cur_capacity)
        if batch_size <= 0:
            return None, None
        obs, actions, *_ = self.memory.sample_batch(batch_size, device=self.device)
        obs = {k: v.detach() for k, v in obs.items()}
        actions = actions.detach()
        return obs, actions

    def train_discriminator(self):
        self.discriminator.train()
        losses = []
        for _ in range(self.disc_iters):
            pol_obs, pol_act = self._sample_policy_batch(self.policy_batch_size)
            if pol_obs is None or pol_act is None:
                break
            exp_obs, exp_act = self._sample_expert_batch(self.expert_batch_size)

            pol_logits = self.discriminator(pol_obs, pol_act)
            exp_logits = self.discriminator(exp_obs, exp_act)

            real_label = 1.0 - self.label_smooth
            fake_label = 0.0 + self.label_smooth
            loss_real = F.binary_cross_entropy_with_logits(exp_logits, torch.full_like(exp_logits, real_label))
            loss_fake = F.binary_cross_entropy_with_logits(pol_logits, torch.full_like(pol_logits, fake_label))
            loss = loss_real + loss_fake

            self.disc_optim.zero_grad(set_to_none=True)
            loss.backward()
            self.disc_optim.step()
            losses.append(loss.detach())

        self.discriminator.eval()
        if len(losses) == 0:
            return {"disc_loss": torch.tensor(0.0, device=self.device)}
        return {"disc_loss": torch.stack(losses).mean()}

    def train(self):
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.ones((self.num_actors,), dtype=torch.bool, device=self.device)

        self.set_eval()
        trajectory, steps = self.explore_env(self.env, self.sac_config.warm_up, random=True)
        self.memory.add_to_buffer(trajectory)
        self.agent_steps += steps

        disc_metrics = {"disc_loss": torch.tensor(0.0, device=self.device)}

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            self.set_eval()
            trajectory, steps = self.explore_env(self.env, self.sac_config.horizon_len, sample=True)
            self.agent_steps += steps
            self.memory.add_to_buffer(trajectory)

            disc_metrics = self.train_discriminator()

            self.set_train()
            results = self.update_net(self.memory)

            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics.update(
                {
                    "epoch": self.epoch,
                    "mini_epoch": self.mini_epoch,
                    "alpha": self.get_alpha(scalar=True),
                    "disc_loss": disc_metrics["disc_loss"].item(),
                }
            )
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            timings = self.timer.stats(step=self.agent_steps, total_names=self.timer_total_names, reset=False)
            timing_metrics = {f"train_timings/{k}": v for k, v in timings.items()}
            metrics.update(timing_metrics)

            episode_metrics = {
                "train_scores/episode_rewards": self.metrics.episode_trackers["rewards"].mean(),
                "train_scores/episode_lengths": self.metrics.episode_trackers["lengths"].mean(),
                "train_scores/num_episodes": self.metrics.num_episodes,
                **self.metrics.result(prefix="train"),
            }
            metrics.update(episode_metrics)

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            self._checkpoint_save(metrics["train_scores/episode_rewards"])

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(
                    f"Epochs: {self.epoch + 1} |",
                    f"Agent Steps: {int(self.agent_steps):,} |",
                    f'Best: {self.best_stat if self.best_stat is not None else -float("inf"):.2f} |',
                    "Stats:",
                    f'ep_rewards {episode_metrics["train_scores/episode_rewards"]:.2f},',
                    f'ep_lengths {episode_metrics["train_scores/episode_lengths"]:.2f},',
                    f'last_sps {timings["lastrate"]:.2f},',
                    f'SPS {timings["totalrate"]:.2f} |',
                    f'disc_loss {disc_metrics["disc_loss"].item():.4f}',
                )

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, "final.pth"))

    def set_train(self):
        super().set_train()
        self.discriminator.train()

    def set_eval(self):
        super().set_eval()
        self.discriminator.eval()
