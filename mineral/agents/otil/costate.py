import collections
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from ..diffrl.utils import adaptive_scheduler, grad_norm, policy_kl
from .otil import OTIL


class Costate(OTIL):
    """Costate-guided actor updates over differentiable dynamics.

    The rollout is differentiated only with respect to detached action leaves.
    The resulting action gradients are then used as local targets for the
    Gaussian actor on detached observations.
    """

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        self.critic_disabled = True
        self.costate_config = full_cfg.agent.get("costate", {})
        self.costate_grad_clip = self.costate_config.get("costate_grad_clip", None)
        self.costate_grad_normalize = self.costate_config.get("costate_grad_normalize", False)
        self.costate_loss_scale = self.costate_config.get("loss_scale", 1.0)
        self.costate_use_reparameterized_surrogate = self.costate_config.get(
            "use_reparameterized_surrogate",
            True,
        )

    def train(self):
        """Actor-only training loop driven by local costate targets."""
        self.initialize_env()

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            if self.max_epochs > 0 and self.epoch >= self.max_epochs:
                print("Reached max epochs. Exiting cleanly.")
                break
            if self.job_clock is not None:
                _, safe_stop = self.job_clock.step(check_safe_stop=True)
                if safe_stop:
                    print("Not enough time left for another step. Exiting cleanly.")
                    break

            if self.shac_config.lr_schedule == "linear":
                assert self.max_epochs > 0
                actor_lr = (self.min_lr - self.actor_lr) * float(self.epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optim.param_groups:
                    param_group["lr"] = actor_lr
                lr = actor_lr
            elif self.shac_config.lr_schedule == "constant":
                lr = self.actor_lr
            elif self.shac_config.lr_schedule == "kl":
                if self.avg_kl is not None:
                    actor_lr = adaptive_scheduler(self.last_lr, self.avg_kl.item(), **self.scheduler_kwargs)
                    for param_group in self.actor_optim.param_groups:
                        param_group["lr"] = actor_lr
                    self.last_lr = actor_lr
                lr = self.last_lr
            else:
                raise NotImplementedError(self.shac_config.lr_schedule)

            self.timer.start("train/update_actor")
            self.actor_encoder.train()
            self.actor.train()
            actor_results = self.update_actor()
            self.timer.end("train/update_actor")

            actor_metrics = {
                k: torch.mean(torch.stack(v)).item() for k, v in actor_results.items() if k not in ["mu", "sigma"]
            }
            actor_metrics.update({k: torch.mean(torch.cat(actor_results[k]), 0).cpu().numpy() for k in ["mu", "sigma"]})
            actor_metrics.update({"epoch": self.epoch, "lr": lr})

            timings = self.timer.stats(step=self.agent_steps, total_names=("train/update_actor",), reset=False)
            metrics = {f"train_stats/{k}": v for k, v in actor_metrics.items()}
            metrics.update({f"train_timings/{k}": v for k, v in timings.items()})

            should_checkpoint = len(self.episode_rewards_hist) > 0
            if should_checkpoint:
                mean_episode_rewards = self.episode_rewards_tracker.mean()
                mean_episode_lengths = self.episode_lengths_tracker.mean()
                mean_episode_discounted_rewards = self.episode_discounted_rewards_tracker.mean()
                metrics.update(
                    {
                        "train_scores/num_episodes": self.num_episodes.item(),
                        "train_scores/episode_rewards": mean_episode_rewards,
                        "train_scores/episode_lengths": mean_episode_lengths,
                        "train_scores/episode_discounted_rewards": mean_episode_discounted_rewards,
                    },
                )
            else:
                mean_episode_rewards = -np.inf
                mean_episode_lengths = 0
                mean_episode_discounted_rewards = -np.inf

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            if should_checkpoint:
                self._checkpoint_save(mean_episode_rewards)

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(
                    f"Epochs: {self.epoch + 1} |",
                    f"Agent Steps: {int(self.agent_steps):,} |",
                    f"SPS: {timings['lastrate']:.2f} |",
                    f"Best: {self.best_stat if self.best_stat is not None else -float('inf'):.2f} |",
                    "Stats:",
                    f"ep_rewards {mean_episode_rewards:.2f},",
                    f"ep_lengths {mean_episode_lengths:.2f},",
                    f"ep_discounted_rewards {mean_episode_discounted_rewards:.2f},",
                    f"costate_norm {metrics['train_stats/costate_grad_norm_mean']:.2f},",
                    f"grad_norm_before_clip/actor {metrics['train_stats/grad_norm_before_clip/actor']:.2f},",
                    f"grad_norm_after_clip/actor {metrics['train_stats/grad_norm_after_clip/actor']:.2f},",
                    "\b\b |",
                )

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, "final.pth"))
        self.episode_rewards_hist = np.array(self.episode_rewards_hist)
        self.episode_lengths_hist = np.array(self.episode_lengths_hist)
        self.episode_discounted_rewards_hist = np.array(self.episode_discounted_rewards_hist)
        np.save(open(os.path.join(self.logdir, "ep_rewards_hist.npy"), "wb"), self.episode_rewards_hist)
        np.save(open(os.path.join(self.logdir, "ep_lengths_hist.npy"), "wb"), self.episode_lengths_hist)
        np.save(
            open(os.path.join(self.logdir, "ep_discounted_rewards_hist.npy"), "wb"),
            self.episode_discounted_rewards_hist,
        )

    def update_actor(self):
        results = collections.defaultdict(list)

        with torch.no_grad():
            self.action_buf.zero_()
            self.mus.zero_()
            self.sigmas.zero_()
            self.rew_buf.zero_()
            self.done_mask.zero_()
            self.next_values.zero_()
            self.avg_next_values.zero_()
            self.target_values.zero_()

        rollout = self.compute_costate_targets()

        def actor_closure():
            self.actor_optim.zero_grad()

            self.timer.start("train/actor_closure/local_surrogate")
            actor_loss, surrogate_info = self.compute_local_actor_loss(rollout)
            self.timer.end("train/actor_closure/local_surrogate")

            actor_loss.backward()

            with torch.no_grad():
                grad_norm_before_clip = grad_norm(self.actor.parameters())
                if self.shac_config.truncate_grads:
                    if self.shac_config.get("max_grad_value", None) is not None:
                        nn.utils.clip_grad_value_(self.actor.parameters(), self.shac_config.max_grad_value)
                    elif self.shac_config.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.shac_config.max_grad_norm)
                grad_norm_after_clip = grad_norm(self.actor.parameters())

                if torch.isnan(grad_norm_before_clip) or grad_norm_before_clip > 1e6:
                    print("NaN gradient - skipping update", grad_norm_before_clip)
                    self.actor_optim.zero_grad()

            for key, val in surrogate_info.items():
                results[key].append(val.detach() if torch.is_tensor(val) else torch.tensor(val))
            results["actor_loss"].append(actor_loss.detach())
            results["grad_norm_before_clip/actor"].append(grad_norm_before_clip)
            results["grad_norm_after_clip/actor"].append(grad_norm_after_clip)
            return actor_loss

        self.actor_optim.step(actor_closure)

        info = rollout["info"]
        stat_keys = (
            "Lk_min",
            "Lk_mean",
            "pseudo_reward_mean",
            "pseudo_reward_std",
            "imitation_loss",
            "returns_mean",
            "expert_return",
            "per_step_cost_mean",
            "per_step_cost_std",
            "costate_grad_norm_mean",
            "costate_grad_norm_max",
            "costate_grad_norm_std",
        )
        for key in stat_keys:
            if key in info:
                val = info[key]
                results[key].append(val.detach() if torch.is_tensor(val) else torch.tensor(val, device=self.device))

        with torch.no_grad():
            obs = {k: v.view(-1, *v.shape[2:]) for k, v in self.obs_buf.items()}
            _, mu, sigma, _ = self.get_actions(obs, sample=False, dist=True)
            old_mu = self.mus.view(-1, self.num_actions)
            old_sigma = self.sigmas.view(-1, self.num_actions)

            kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)
            results["mu"].append(mu)
            results["sigma"].append(sigma)
            kl_dist /= self.num_actions
            avg_kl = kl_dist.mean()
            results["avg_kl"].append(avg_kl)
            self.avg_kl = avg_kl

        return results

    def compute_costate_targets(self):
        with torch.no_grad():
            obs_rms = deepcopy(self.obs_rms) if self.obs_rms is not None else None

        obs = self.env.initialize_trajectory()
        obs = self._convert_obs(obs)

        obs_window = {k: [] for k in obs.keys()}
        if self.input_type == "state_state":
            obs_window = {k: [v] for k, v in obs.items()}

        stored_obs = []
        action_leaves = []
        action_noises = []

        if obs_rms is not None:
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_rms[k].update(v)
            obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

        for i in range(self.horizon_len):
            obs_for_actor = {k: v.detach() for k, v in obs.items()}
            stored_obs.append(obs_for_actor)

            with torch.no_grad():
                for k, v in obs_for_actor.items():
                    self.obs_buf[k][i] = v.clone()

                z = self.actor_encoder(obs_for_actor)
                policy_actions, mu, sigma, _ = self.get_actions(obs_for_actor, z=z, sample=True, dist=True)
                if self.costate_use_reparameterized_surrogate:
                    denom = sigma.detach().clamp_min(1e-6)
                    action_noises.append(((policy_actions - mu.detach()) / denom).detach())
                else:
                    action_noises.append(None)

                self.mus[i, ...] = mu.clone()
                self.sigmas[i, ...] = sigma.clone()

            action_leaf = policy_actions.detach().requires_grad_(True)
            action_leaves.append(action_leaf)
            with torch.no_grad():
                self.action_buf[i] = action_leaf.detach()

            self.timer.start("train/compute_costate_targets/forward_sim")
            obs, rew, done, info = self.env.step(action_leaf)
            self.timer.end("train/compute_costate_targets/forward_sim")

            real_obs = info.get("obs_before_reset", obs)
            real_obs = obs if real_obs is None else real_obs
            obs = self._convert_obs(obs)
            real_obs = self._convert_obs(real_obs)

            with torch.no_grad():
                raw_rew = rew.clone()
                self.episode_rewards += raw_rew
                self.episode_discounted_rewards += self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                self.episode_lengths += 1

            if obs_rms is not None:
                with torch.no_grad():
                    for k, v in obs.items():
                        self.obs_rms[k].update(v)
                obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}
                real_obs = {k: obs_rms[k].normalize(v) for k, v in real_obs.items()}

            for k, v in real_obs.items():
                obs_window[k].append(v)

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                self._record_done_episodes(done_env_ids, obs, info, obs_rms)

            if i < self.horizon_len - 1:
                self.done_mask[i] = done.to(dtype=torch.float32)
            else:
                self.done_mask[i] = torch.ones_like(done, dtype=torch.float32)

        obs_window = {k: torch.stack(v, dim=1) for k, v in obs_window.items()}
        with torch.no_grad():
            obs_exp = {k: obs_rms[k].normalize(v) for k, v in self.demos["obs"].items()} if obs_rms is not None else self.demos["obs"]

        obs_z = self.actor_encoder(obs_window)
        exp_z = self.actor_encoder(obs_exp).detach()

        self.timer.start("train/compute_costate_targets/loss_fn")
        imitation_loss, info = self.loss_fn(obs_z, exp_z, sim_is_window=True)
        self.timer.end("train/compute_costate_targets/loss_fn")

        if "per_step_costs" not in info:
            raise ValueError("Costate requires loss info with per_step_costs")
        self._maybe_save_plan_heatmaps(info)
        per_step_costs = info.pop("per_step_costs")
        per_step_rewards = self._build_pseudo_rewards(per_step_costs)
        step_rewards = per_step_rewards.transpose(0, 1)

        with torch.no_grad():
            self.rew_buf.copy_(step_rewards.detach())

        returns = self._compute_discounted_returns(step_rewards)
        objective_loss = -returns.mean()

        self.timer.start("train/compute_costate_targets/backward_costate")
        action_grads = torch.autograd.grad(
            objective_loss,
            action_leaves,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        self.timer.end("train/compute_costate_targets/backward_costate")

        action_grads = torch.stack([g.detach() for g in action_grads], dim=0)
        action_grads = torch.nan_to_num(action_grads, nan=0.0, posinf=0.0, neginf=0.0)

        if self.costate_grad_normalize:
            grad_scale = action_grads.norm(dim=-1, keepdim=True).mean().clamp_min(1e-6)
            action_grads = action_grads / grad_scale

        if self.costate_grad_clip is not None:
            grad_norms = action_grads.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            clip = torch.as_tensor(self.costate_grad_clip, device=self.device, dtype=action_grads.dtype)
            action_grads = action_grads * torch.clamp(clip / grad_norms, max=1.0)

        grad_norms = action_grads.norm(dim=-1)
        info["pseudo_reward_mean"] = per_step_rewards.detach().mean()
        info["pseudo_reward_std"] = per_step_rewards.detach().std(unbiased=False)
        info["imitation_loss"] = imitation_loss.detach()
        info["per_step_cost_mean"] = per_step_costs.detach().mean()
        info["per_step_cost_std"] = per_step_costs.detach().std(unbiased=False)
        info["returns_mean"] = returns.detach().mean()
        info["expert_return"] = self.demos["expert_return"]
        info["costate_grad_norm_mean"] = grad_norms.mean()
        info["costate_grad_norm_max"] = grad_norms.max()
        info["costate_grad_norm_std"] = grad_norms.std(unbiased=False)

        self.agent_steps += self.horizon_len * self.num_envs
        return {
            "obs": stored_obs,
            "action_grads": action_grads,
            "action_noises": action_noises,
            "info": info,
        }

    def compute_local_actor_loss(self, rollout):
        surrogate_terms = []
        target_norms = []

        for i, obs in enumerate(rollout["obs"]):
            z = self.actor_encoder(obs)
            _, mu, sigma, _ = self.get_actions(obs, z=z, sample=False, dist=True)

            noise = rollout["action_noises"][i]
            if self.costate_use_reparameterized_surrogate and noise is not None:
                action_pred = mu + sigma * noise
                if self.tanh_clamp:
                    action_pred = torch.tanh(action_pred)
            else:
                action_pred = mu

            target_grad = rollout["action_grads"][i]
            surrogate_terms.append((action_pred * target_grad).sum(dim=-1))
            target_norms.append(target_grad.norm(dim=-1))

        surrogate = torch.stack(surrogate_terms, dim=0)
        target_norms = torch.stack(target_norms, dim=0)
        actor_loss = self.costate_loss_scale * surrogate.mean()
        info = {
            "local_surrogate": surrogate.detach().mean(),
            "target_grad_norm_mean": target_norms.detach().mean(),
        }
        return actor_loss, info

    def _compute_discounted_returns(self, step_rewards):
        returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        rew_acc = torch.zeros_like(returns)
        gamma = torch.ones_like(returns)
        for i in range(self.horizon_len):
            rew_acc = rew_acc + gamma * step_rewards[i]
            done = self.done_mask[i]
            returns = returns + rew_acc * done
            gamma = gamma * self.gamma
            not_done = 1.0 - done
            rew_acc = rew_acc * not_done
            gamma = gamma * not_done + torch.ones_like(gamma) * done
        return returns

    def _record_done_episodes(self, done_env_ids, obs, info, obs_rms):
        done_env_ids = done_env_ids.detach().cpu()
        terminal_obs = info.get("obs_before_reset", None)
        terminal_obs = obs if terminal_obs is None else self._convert_obs(terminal_obs)
        if obs_rms is not None:
            terminal_obs = {k: obs_rms[k].normalize(v) for k, v in terminal_obs.items()}

        self.episode_rewards_tracker.update(self.episode_rewards[done_env_ids])
        self.episode_discounted_rewards_tracker.update(self.episode_discounted_rewards[done_env_ids])
        self.episode_lengths_tracker.update(self.episode_lengths[done_env_ids])
        self.num_episodes += len(done_env_ids)

        for done_env_id in done_env_ids:
            self.episode_rewards_hist.append(self.episode_rewards[done_env_id].item())
            self.episode_discounted_rewards_hist.append(self.episode_discounted_rewards[done_env_id].item())
            self.episode_lengths_hist.append(self.episode_lengths[done_env_id].item())
            self.episode_rewards[done_env_id] = 0.0
            self.episode_discounted_rewards[done_env_id] = 0.0
            self.episode_lengths[done_env_id] = 0
            self.episode_gamma[done_env_id] = 1.0
