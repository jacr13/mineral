import collections
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from ...common.demos import get_demos
from ..diffrl.shac import SHAC
from ..diffrl.utils import adaptive_scheduler, grad_norm, policy_kl
from .chamfer_loss import ChamferImitationLoss


class ILD(SHAC):
    """Differentiable ILD that optimizes a Chamfer imitation loss directly."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        self.ild_config = full_cfg.agent.get("ild", {})

        demos_config = self.ild_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)

        self.loss_fn = ChamferImitationLoss()

    def train(self):
        """Actor-only training loop (no critic/bootstrap)."""
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
            self.critic.eval()
            self.critic_target.eval()
            actor_results = self.update_actor()
            self.timer.end("train/update_actor")

            actor_metrics = {
                k: torch.mean(torch.stack(v)).item() for k, v in actor_results.items() if k not in ["mu", "sigma"]
            }
            actor_metrics.update({k: torch.mean(torch.cat(actor_results[k]), 0).cpu().numpy() for k in ["mu", "sigma"]})
            actor_metrics.update({"epoch": self.epoch, "lr": lr})

            timings = self.timer.stats(step=self.agent_steps, total_names=("train/update_actor",), reset=False)
            timing_metrics = {f"train_timings/{k}": v for k, v in timings.items()}

            metrics = {f"train_stats/{k}": v for k, v in actor_metrics.items()}
            metrics.update(timing_metrics)

            should_checkpoint = len(self.episode_rewards_hist) > 0
            if should_checkpoint:
                mean_episode_rewards = self.episode_rewards_tracker.mean()
                mean_episode_lengths = self.episode_lengths_tracker.mean()
                episode_metrics = {
                    "train_scores/num_episodes": self.num_episodes.item(),
                    "train_scores/episode_rewards": mean_episode_rewards,
                    "train_scores/episode_lengths": mean_episode_lengths,
                }
                metrics.update(episode_metrics)
            else:
                mean_episode_rewards = -np.inf

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
                    f"grad_norm_before_clip/actor {metrics['train_stats/grad_norm_before_clip/actor']:.2f},",
                    f"grad_norm_after_clip/actor {metrics['train_stats/grad_norm_after_clip/actor']:.2f},",
                    "\b\b |",
                )

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, "final.pth"))
        self.episode_rewards_hist = np.array(self.episode_rewards_hist)
        self.episode_lengths_hist = np.array(self.episode_lengths_hist)
        np.save(open(os.path.join(self.logdir, "ep_rewards_hist.npy"), "wb"), self.episode_rewards_hist)
        np.save(open(os.path.join(self.logdir, "ep_lengths_hist.npy"), "wb"), self.episode_lengths_hist)

    def update_actor(self):
        results = collections.defaultdict(list)

        with torch.no_grad():
            self.action_buf.zero_()
            self.mus.zero_()
            self.sigmas.zero_()
            self.done_mask.zero_()

        def actor_closure():
            self.actor_optim.zero_grad()

            self.timer.start("train/actor_closure/forward_sim")
            actor_loss, info = self.compute_actor_loss()
            self.timer.end("train/actor_closure/forward_sim")

            self.timer.start("train/actor_closure/backward_sim")
            actor_loss.backward()
            self.timer.end("train/actor_closure/backward_sim")

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

            stat_keys = (
                "imitation_loss",
                "loss_deviation",
                "loss_coverage",
            )
            for key in stat_keys:
                if key in info:
                    val = info[key]
                    if torch.is_tensor(val):
                        val = val.detach()
                    else:
                        val = torch.tensor(val)
                    results[key].append(val)

            results["actor_loss"].append(actor_loss.detach())
            results["grad_norm_before_clip/actor"].append(grad_norm_before_clip)
            results["grad_norm_after_clip/actor"].append(grad_norm_after_clip)
            return actor_loss

        self.actor_optim.step(actor_closure)

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

    def compute_actor_loss(self):
        with torch.no_grad():
            obs_rms = deepcopy(self.obs_rms) if self.obs_rms is not None else None

        obs = self.env.initialize_trajectory()
        obs = self._convert_obs(obs)

        obs_window = {k: [] for k in obs.keys()}

        if obs_rms is not None:
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_rms[k].update(v)
            obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

        for i in range(self.horizon_len):
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_buf[k][i] = v.clone()

            z = self.actor_encoder(obs)
            actions, mu, sigma, distr = self.get_actions(obs, z=z, sample=True, dist=True)

            with torch.no_grad():
                self.action_buf[i] = actions.clone()
                self.mus[i, ...] = mu.clone()
                self.sigmas[i, ...] = sigma.clone()

            obs, rew, done, info = self.env.step(actions)
            real_obs = info.get("obs_before_reset", obs)
            real_obs = obs if real_obs is None else real_obs

            obs = self._convert_obs(obs)
            real_obs = self._convert_obs(real_obs)

            with torch.no_grad():
                self.episode_rewards += rew
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
                done_env_ids = done_env_ids.detach().cpu()
                terminal_obs = info.get("obs_before_reset", None)
                terminal_obs = obs if terminal_obs is None else terminal_obs
                terminal_obs = self._convert_obs(terminal_obs)
                if obs_rms is not None:
                    terminal_obs = {k: obs_rms[k].normalize(v) for k, v in terminal_obs.items()}
                self.episode_rewards_tracker.update(self.episode_rewards[done_env_ids])
                self.episode_lengths_tracker.update(self.episode_lengths[done_env_ids])
                self.num_episodes += len(done_env_ids)
                for done_env_id in done_env_ids:
                    self.episode_rewards_hist.append(self.episode_rewards[done_env_id].item())
                    self.episode_lengths_hist.append(self.episode_lengths[done_env_id].item())
                    self.episode_rewards[done_env_id] = 0.0
                    self.episode_lengths[done_env_id] = 0

            if i < self.horizon_len - 1:
                self.done_mask[i] = done.to(dtype=torch.float32)
            else:
                self.done_mask[i] = torch.ones_like(done, dtype=torch.float32)

        obs_window = {k: torch.stack(v, dim=1) for k, v in obs_window.items()}
        if obs_rms is not None:
            with torch.no_grad():
                obs_exp = {k: obs_rms[k].normalize(v) for k, v in self.demos["obs"].items()}
        else:
            obs_exp = self.demos["obs"]

        obs_z = self.actor_encoder(obs_window)
        exp_z = self.actor_encoder(obs_exp).detach()
        actor_loss, info = self.loss_fn(obs_z, exp_z)
        info["imitation_loss"] = actor_loss.detach()
        info["expert_return"] = self.demos["expert_return"]

        self.agent_steps += self.horizon_len * self.num_envs
        return actor_loss, info
