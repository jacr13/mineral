import collections
import os

import torch
import torch.nn.functional as F

from ...common.demos import get_demos
from ..ddpg.models import InverseModel
from ..ddpg.utils import soft_update
from ..gail.models import Discriminator
from ..sac.sac import SAC


class DAC(SAC):
    """Discriminator Actor-Critic (imitation learning) built on SAC."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)

        self.dac_config = full_cfg.agent.get("dac", {})
        self.input_type = self.dac_config.get("input_type", "state_action")

        demos_config = self.dac_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)

        discriminator_config = self.dac_config.get("discriminator", {})
        if "mlp_kwargs" not in discriminator_config:
            mlp_kwargs = self.dac_config.get("discriminator_mlp", None)
            if mlp_kwargs is not None:
                discriminator_config = dict(discriminator_config)
                discriminator_config["mlp_kwargs"] = mlp_kwargs

        encoder_kwargs = discriminator_config.get("encoder_kwargs", None)
        if encoder_kwargs is None:
            encoder_kwargs = self.network_config.get("encoder_kwargs", {})

        self.discriminator = Discriminator(
            self.obs_space,
            self.action_dim,
            input_type=self.input_type,
            discriminator_kwargs=discriminator_config,
            encoder_kwargs=encoder_kwargs,
        ).to(self.device)

        disc_optim_kwargs = (
            self.dac_config.get("discriminator_optim")
            or discriminator_config.get("optim")
            or {"type": "Adam", "kwargs": {"lr": 3e-4}}
        )
        DiscOptim = getattr(torch.optim, disc_optim_kwargs.get("type", "Adam"))
        self.disc_optim = DiscOptim(self.discriminator.parameters(), **disc_optim_kwargs.get("kwargs", {}))

        self.disc_iters = int(self.dac_config.get("discriminator_iters", discriminator_config.get("iters", 1)))
        self.expert_batch_size = int(self.dac_config.get("expert_batch_size", self.sac_config.batch_size))
        self.policy_batch_size = int(self.dac_config.get("policy_batch_size", self.sac_config.batch_size))
        self.reward_scale = float(self.dac_config.get("reward_scale", 1.0))
        self.label_smooth = float(self.dac_config.get("label_smooth", 0.0))
        self.disc_update_freq = int(self.dac_config.get("disc_update_freq", 1))

        inv_config = self.dac_config.get("inverse_model", {})
        self.inv_reg_coef = float(inv_config.get("reg_coef", 0.0))
        self.inv_update_freq = int(inv_config.get("update_freq", 1))
        self.inv_batch_size = int(inv_config.get("batch_size", self.sac_config.batch_size))
        self.inv_train_iters = int(inv_config.get("iters", 1))
        inv_patience = inv_config.get("patience", 0)
        self.inv_patience = 0 if inv_patience is None else int(inv_patience)
        self.inv_min_delta = float(inv_config.get("min_delta", 0.0))
        self.inverse_model = None
        self.inv_optim = None
        if inv_config.get("enabled", self.inv_reg_coef > 0.0):
            inv_mlp_kwargs = inv_config.get("mlp_kwargs", None)
            inv_encoder_kwargs = inv_config.get("encoder_kwargs", None)
            if inv_encoder_kwargs is None:
                inv_encoder_kwargs = self.network_config.get("encoder_kwargs", {})
            self.inverse_model = InverseModel(
                obs_space=self.obs_space,
                action_dim=self.action_dim,
                mlp_kwargs=inv_mlp_kwargs,
                encoder_kwargs=inv_encoder_kwargs,
            ).to(self.device)
            inv_optim_kwargs = inv_config.get("optim", {"type": "Adam", "kwargs": {"lr": 3e-4}})
            InvOptim = getattr(torch.optim, inv_optim_kwargs.get("type", "Adam"))
            self.inv_optim = InvOptim(self.inverse_model.parameters(), **inv_optim_kwargs.get("kwargs", {}))

        self.discriminator.eval()
        if self.inverse_model is not None:
            self.inverse_model.eval()

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
    def dac_reward(self, obs, actions, next_obs):
        input_1, input_2 = self.get_discriminator_inputs(obs, actions, next_obs)
        logits = self.discriminator(input_1, input_2)
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

        for i in range(timesteps):
            if not self.env_autoresets:
                raise NotImplementedError

            if self.normalize_input:
                for k, v in self.obs.items():
                    self.obs_rms[k].update(v)
            if random:
                actions = torch.rand((self.num_actors, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                actions = self.get_actions(obs=self.obs, sample=sample)

            next_obs, rewards, dones, infos = env.step(actions)
            next_obs = self._convert_obs(next_obs)

            done_indices = torch.where(dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards, done_indices, infos)

            if self.sac_config.handle_timeout:
                dones = self._handle_timeout(dones, infos)

            shaped_rewards = self.dac_reward(self.obs, actions, next_obs).squeeze(-1)

            for k, v in self.obs.items():
                traj_obs[k][:, i] = v
            traj_actions[:, i] = actions
            traj_dones[:, i] = dones
            traj_rewards[:, i] = shaped_rewards
            for k, v in next_obs.items():
                traj_next_obs[k][:, i] = v
            self.obs = next_obs

        self.metrics.flush_video(self.epoch)

        traj_rewards = self.reward_shaper(traj_rewards.reshape(self.num_actors, timesteps, 1))
        traj_dones = traj_dones.reshape(self.num_actors, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones)
        return data, timesteps * self.num_actors

    def _sample_batch(self, obs_rollout, actions, batch_size, next_obs_rollout=None, dones=None):
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(0)
            obs_rollout = {k: v.unsqueeze(0) for k, v in obs_rollout.items()}
        num_trajs, length, _ = actions.shape

        if length < 2:
            raise ValueError("Rollout length must be >= 2 to sample next_obs")

        if dones is not None:
            if len(dones.shape) == 1:
                dones = dones.unsqueeze(0)
            assert dones.shape[:2] == (num_trajs, length), f"dones must have shape ({num_trajs}, {length}), got {dones.shape}"

            valid_mask = ~dones[:, :-1]
            valid_indices = valid_mask.nonzero(as_tuple=False)

            if len(valid_indices) == 0:
                raise ValueError("No valid (obs, next_obs) pairs found before dones.")

            sample_idx = torch.randint(0, len(valid_indices), (batch_size,), device=self.device)
            trajs_idx = valid_indices[sample_idx, 0]
            steps_idx = valid_indices[sample_idx, 1]
        else:
            trajs_idx = torch.randint(0, num_trajs, (batch_size,), device=self.device)
            steps_idx = torch.randint(0, length - 1, (batch_size,), device=self.device)

        obs = {k: v[trajs_idx, steps_idx, ...].detach() for k, v in obs_rollout.items()}
        act = actions[trajs_idx, steps_idx, :].detach()
        if next_obs_rollout is None:
            next_obs = {k: v[trajs_idx, steps_idx + 1, ...].detach() for k, v in obs_rollout.items()}
        else:
            next_obs = {k: v[trajs_idx, steps_idx, ...].detach() for k, v in next_obs_rollout.items()}

        if self.normalize_input:
            obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
            next_obs = {k: self.obs_rms[k].normalize(v) for k, v in next_obs.items()}

        return obs, act, next_obs

    def _normalize_obs_dict(self, obs):
        if self.normalize_input:
            return {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        return obs

    def train_discriminator(self, memory):
        if memory.cur_capacity < self.policy_batch_size:
            return {
                "discriminator/total": 0.0,
                "discriminator/real": 0.0,
                "discriminator/fake": 0.0,
            }

        self.discriminator.train()
        losses = {"total": [], "real": [], "fake": []}

        for _ in range(self.disc_iters):
            pol_obs, pol_act, _, pol_next_obs, _ = memory.sample_batch(self.policy_batch_size, device=self.device)
            if self.normalize_input:
                pol_obs = {k: self.obs_rms[k].normalize(v) for k, v in pol_obs.items()}
                pol_next_obs = {k: self.obs_rms[k].normalize(v) for k, v in pol_next_obs.items()}

            exp_obs, exp_act, exp_next_obs = self._sample_batch(
                self.demos["obs"],
                self.demos["act"],
                self.expert_batch_size,
                dones=self.demos["done"],
                next_obs_rollout=self.demos["next_obs"],
            )

            pol_input_1, pol_input_2 = self.get_discriminator_inputs(pol_obs, pol_act, pol_next_obs)
            pol_logits = self.discriminator(pol_input_1, pol_input_2)

            exp_input_1, exp_input_2 = self.get_discriminator_inputs(exp_obs, exp_act, exp_next_obs)
            exp_logits = self.discriminator(exp_input_1, exp_input_2)

            real_label = 1.0 - self.label_smooth
            fake_label = 0.0 + self.label_smooth
            loss_real = F.binary_cross_entropy_with_logits(exp_logits, torch.full_like(exp_logits, real_label))
            loss_fake = F.binary_cross_entropy_with_logits(pol_logits, torch.full_like(pol_logits, fake_label))
            loss = loss_real + loss_fake

            self.disc_optim.zero_grad()
            loss.backward()
            self.disc_optim.step()

            losses["total"].append(loss.detach())
            losses["real"].append(loss_real.detach())
            losses["fake"].append(loss_fake.detach())

        self.discriminator.eval()
        return {
            "discriminator/total": torch.stack(losses["total"]).mean().item() if len(losses["total"]) > 0 else 0.0,
            "discriminator/real": torch.stack(losses["real"]).mean().item() if len(losses["real"]) > 0 else 0.0,
            "discriminator/fake": torch.stack(losses["fake"]).mean().item() if len(losses["fake"]) > 0 else 0.0,
        }

    def update_actor(self, obs, next_obs=None):
        self.critic.requires_grad_(False)
        obs = self._normalize_obs_dict(obs)
        z = self.encoder(obs)
        if self.sac_config.get("actor_detach_encoder", False):
            z = {k: v.detach() for k, v in z.items()} if isinstance(z, dict) else z.detach()
        actions, _, log_prob = self.get_actions(z=z, logprob=True)
        Q = self.critic.get_q_min(z, actions)
        actor_loss = (self.get_alpha() * log_prob - Q).mean()

        inv_reg_loss = None
        if self.inverse_model is not None and next_obs is not None and self.inv_reg_coef > 0.0:
            next_obs = self._normalize_obs_dict(next_obs)
            with torch.no_grad():
                inv_pred = self.inverse_model(obs, next_obs)
            inv_reg_loss = F.mse_loss(actions, inv_pred)
            actor_loss = actor_loss + self.inv_reg_coef * inv_reg_loss

        grad_norm = self.optimizer_update(self.actor_optim, actor_loss)
        self.critic.requires_grad_(True)

        entropy = -log_prob
        alpha_loss = None
        if self.sac_config.alpha is None:
            alpha_loss = (self.get_alpha(detach=False) * (entropy - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, alpha_loss)
        return actor_loss, alpha_loss, entropy.mean(), grad_norm, inv_reg_loss

    def update_net(self, memory):
        results = collections.defaultdict(list)
        for _i in range(self.sac_config.mini_epochs):
            self.mini_epoch += 1
            obs, action, reward, next_obs, done = memory.sample_batch(self.sac_config.batch_size, device=self.device)

            critic_loss, critic_grad_norm, target_values = self.update_critic(obs, action, reward, next_obs, done)
            results["loss/critic"].append(critic_loss)
            results["grad_norm/critic"].append(critic_grad_norm)
            for k, v in target_values.items():
                results[k].append(v)

            if self.inverse_model is not None and self.mini_epoch % self.inv_update_freq == 0:
                normalizer = self.obs_rms if self.normalize_input else None
                inv_loss = self.inverse_model.train_on_replay(
                    memory,
                    self.inv_optim,
                    batch_size=self.inv_batch_size,
                    device=self.device,
                    iters=self.inv_train_iters,
                    normalizer=normalizer,
                    patience=self.inv_patience,
                    min_delta=self.inv_min_delta,
                )
                if inv_loss is not None:
                    results["loss/inverse_model"].append(inv_loss)

            if self.mini_epoch % self.sac_config.update_actor_interval == 0:
                actor_loss, alpha_loss, entropy, actor_grad_norm, inv_reg_loss = self.update_actor(obs, next_obs)
                results["loss/actor"].append(actor_loss)
                if alpha_loss is not None:
                    results["loss/alpha"].append(alpha_loss)
                results["entropy"].append(entropy)
                results["grad_norm/actor"].append(actor_grad_norm)
                if inv_reg_loss is not None:
                    results["loss/inv_reg"].append(inv_reg_loss)

            if self.mini_epoch % self.sac_config.update_targets_interval == 0:
                soft_update(self.critic_target, self.critic, self.sac_config.tau)
                if not self.sac_config.no_tgt_actor:
                    soft_update(self.actor_target, self.actor, self.sac_config.tau)
        return results

    def train(self):
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.ones((self.num_actors,), dtype=torch.bool, device=self.device)

        self.set_eval()
        trajectory, steps = self.explore_env(self.env, self.sac_config.warm_up, random=True)
        self.memory.add_to_buffer(trajectory)
        self.agent_steps += steps

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            if self.job_clock is not None:
                _, safe_stop = self.job_clock.step(check_safe_stop=True)
                if safe_stop:
                    print("Not enough time left for another step. Exiting cleanly.")
                    break

            self.set_eval()
            trajectory, steps = self.explore_env(self.env, self.sac_config.horizon_len, sample=True)
            self.agent_steps += steps
            self.memory.add_to_buffer(trajectory)

            disc_metrics = {}
            if self.epoch % self.disc_update_freq == 0:
                disc_metrics = self.train_discriminator(self.memory)

            self.set_train()
            results = self.update_net(self.memory)

            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics.update({"epoch": self.epoch, "mini_epoch": self.mini_epoch, "alpha": self.get_alpha(scalar=True)})
            metrics.update({"expert_return": self.demos["expert_return"]})
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            timings = self.timer.stats(step=self.agent_steps, total_names=self.timer_total_names, reset=False)
            timing_metrics = {f"train_timings/{k}": v for k, v in timings.items()}
            metrics.update(timing_metrics)

            episode_metrics = {
                "train_scores/episode_rewards": self.metrics.episode_trackers["rewards"].mean(),
                "train_scores/episode_lengths": self.metrics.episode_trackers["lengths"].mean(),
                "train_scores/num_episodes": self.metrics.num_episodes,
                **self.metrics.result(prefix="train"),
                **disc_metrics,
            }
            metrics.update(episode_metrics)

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            self._checkpoint_save(metrics["train_scores/episode_rewards"])

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(
                    f"Epochs: {self.epoch + 1} |",
                    f"Agent Steps: {int(self.agent_steps):,} |",
                    f"Best: {self.best_stat if self.best_stat is not None else -float('inf'):.2f} |",
                    "Stats:",
                    f"ep_rewards {episode_metrics['train_scores/episode_rewards']:.2f},",
                    f"ep_lengths {episode_metrics['train_scores/episode_lengths']:.2f},",
                    f"last_sps {timings['lastrate']:.2f},",
                    f"SPS {timings['totalrate']:.2f} |",
                    f"Expert return: {self.demos['expert_return']:.2f}",
                )

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, "final.pth"))
