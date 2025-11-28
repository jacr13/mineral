import collections
import itertools
import json
import os
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import nets
from ...common import normalizers
from ...common.demos import get_demos
from ...common.reward_shaper import RewardShaper
from ...common.timer import Timer
from ...common.tracker import Tracker
from ..agent import Agent
from ..diffrl.utils import CriticDataset, adaptive_scheduler, grad_norm, policy_kl, soft_update
from . import models
from .best_of_k import BestOfK, BestOfKConfig


class OTIL(Agent):
    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.otil_config = full_cfg.agent.otil
        self.num_actors = self.otil_config.num_actors
        self.max_agent_steps = int(self.otil_config.max_agent_steps)
        super().__init__(full_cfg, **kwargs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.max_episode_length

        # --- OTIL Parameters ---
        self.tanh_clamp = self.network_config.get("tanh_clamp", False)  # on actions, if not done in actor dist
        self.actor_detach_z = self.otil_config.get("actor_detach_z", False)
        self.actor_loss_coef = self.otil_config.get("actor_loss_coef", 1.0)
        self.actor_value_coef = self.otil_config.get("actor_value_coef", 0.0)
        self.actor_value_return_type = self.otil_config.get("actor_value_return_type", "min")

        # --- SAPO-style entropy regularization ---
        self.with_logprobs = self.otil_config.get("with_logprobs", False)
        self.with_autoent = self.otil_config.get("with_autoent", False)
        self.entropy_coef = self.otil_config.get("entropy_coef", None)
        self.offset_by_target_entropy = self.otil_config.get("offset_by_target_entropy", False)
        self.scale_by_target_entropy = self.otil_config.get("scale_by_target_entropy", False)
        self.unscale_entropy_alpha = self.otil_config.get("unscale_entropy_alpha", False)
        self.use_distr_ent = self.otil_config.get("use_distr_ent", False)
        self.entropy_in_return = self.otil_config.get("entropy_in_return", False)
        self.entropy_in_targets = self.otil_config.get("entropy_in_targets", False)
        self.track_entropy = (
            self.with_logprobs
            or self.with_autoent
            or (self.entropy_coef is not None)
            or self.entropy_in_return
            or self.entropy_in_targets
        )
        target_entropy_scalar = self.otil_config.get("target_entropy_scalar", 1.0)
        self.target_entropy = -self.action_dim * target_entropy_scalar
        if self.with_autoent or self.entropy_coef is not None:
            print("Target Entropy Scalar:", target_entropy_scalar, "Target Entropy:", self.target_entropy)
        self.log_alpha = None
        self.alpha_optim = None
        self._entropy = None

        self.horizon_len = self.otil_config.horizon_len
        self.max_epochs = self.otil_config.get("max_epochs", 0)  # set to 0 to disable and track by max_agent_steps instead
        self.gamma = self.otil_config.get("gamma", 0.99)
        self.critic_method = self.otil_config.get("critic_method", "one-step")
        if self.critic_method == "td-lambda":
            self.lam = self.otil_config.get("lambda", 0.95)
        self.critic_iterations = self.otil_config.get("critic_iterations", 16)
        self.target_critic_alpha = self.otil_config.get("target_critic_alpha", 0.995)
        self.no_target_critic = self.otil_config.get("no_target_critic", False)
        self.num_critic_batches = self.otil_config.get("num_critic_batches", 4)
        self.critic_batch_size = max(1, self.num_envs * self.horizon_len // max(1, self.num_critic_batches))
        self.critic_reward_reduction = self.otil_config.get("critic_reward_reduction", "min")
        self.critic_reward_scale = self.otil_config.get("critic_reward_scale", 1.0)
        self.critic_reward_tau = self.otil_config.get("critic_reward_tau", 0.5)
        self.critic_reward_normalize = self.otil_config.get("critic_reward_normalize", False)
        self.reward_shaper = RewardShaper(**self.otil_config.get("reward_shaper", {"fn": "scale", "scale": 1.0}))

        # demos
        demos_config = self.otil_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)

        # --- Normalizers ---
        if self.tanh_clamp:  # legacy
            # unbiased=False -> correction=0
            # https://github.com/NVlabs/DiffRL/blob/a4c0dd1696d3c3b885ce85a3cb64370b580cb913/utils/running_mean_std.py#L34
            rms_config = {"eps": 1e-5, "correction": 0, "initial_count": 1e-4, "dtype": torch.float32}
        else:
            rms_config = {"eps": 1e-5, "initial_count": 1, "dtype": torch.float64}
        if self.normalize_input:
            self.obs_rms = {}
            for k, v in self.obs_space.items():
                if re.match(self.obs_rms_keys, k):
                    self.obs_rms[k] = normalizers.RunningMeanStd(v, **rms_config)
                else:
                    self.obs_rms[k] = normalizers.Identity()
            self.obs_rms = nn.ModuleDict(self.obs_rms).to(self.device)
        else:
            self.obs_rms = None

        # --- Encoder ---
        if self.network_config.get("encoder", None) is not None:
            EncoderCls = getattr(nets, self.network_config.encoder)
            encoder_kwargs = self.network_config.get("encoder_kwargs", {})
            self.encoder = EncoderCls(self.obs_space, encoder_kwargs, weight_init_fn=models.weight_init_)
        else:
            f = lambda x: x["obs"]
            self.encoder = nets.Lambda(f)
        self.encoder.to(self.device)
        print("Encoder:", self.encoder)

        self.share_encoder = self.otil_config.get("share_encoder", True)
        if self.share_encoder:
            self.actor_encoder = self.encoder
            print("Actor Encoder: (shared)")
        else:
            self.actor_encoder = deepcopy(self.encoder)
            print("Actor Encoder:", self.actor_encoder)

        # --- Model ---
        if self.network_config.get("encoder", None) is not None:
            obs_dim = self.encoder.out_dim
        else:
            obs_dim = self.obs_space["obs"]
            obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
            assert obs_dim == self.env.num_obs
            assert self.action_dim == self.env.num_actions

        ActorCls = getattr(models, self.network_config.actor)
        self.actor = ActorCls(obs_dim, self.action_dim, **self.network_config.get("actor_kwargs", {}))
        self.actor.to(self.device)
        print("Actor:", self.actor)

        CriticCls = getattr(models, self.network_config.critic)
        self.critic = CriticCls(obs_dim, self.action_dim, **self.network_config.get("critic_kwargs", {}))
        self.critic.to(self.device)
        if self.no_target_critic:
            self.encoder_target = self.encoder
            self.critic_target = self.critic
        else:
            self.encoder_target = deepcopy(self.encoder).to(self.device)
            self.critic_target = deepcopy(self.critic).to(self.device)
        print("Critic:", self.critic)

        # --- Optim ---
        OptimCls = getattr(torch.optim, self.otil_config.optim_type)

        if self.otil_config.get("actor_detach_encoder", False):
            actor_optim_params = self.actor.parameters()
        else:
            actor_optim_params = itertools.chain(self.actor_encoder.parameters(), self.actor.parameters())
        self.actor_optim = OptimCls(
            actor_optim_params,
            **self.otil_config.get("actor_optim_kwargs", {}),
        )
        print("Actor Optim:", self.actor_optim)

        if self.otil_config.get("critic_detach_encoder", False):
            critic_optim_params = self.critic.parameters()
        else:
            critic_optim_params = itertools.chain(self.encoder.parameters(), self.critic.parameters())
        self.critic_optim = OptimCls(
            critic_optim_params,
            **self.otil_config.get("critic_optim_kwargs", {}),
        )
        print("Critic Optim:", self.critic_optim)

        if self.with_autoent and self.otil_config.get("alpha", None) is None:
            alpha_optim_type = self.otil_config.get("alpha_optim_type", self.otil_config.optim_type)
            AlphaOptimCls = getattr(torch.optim, alpha_optim_type)
            init_alpha = np.log(self.otil_config.get("init_alpha", 1.0))
            self.log_alpha = nn.Parameter(torch.tensor(init_alpha, device=self.device, dtype=torch.float32))
            self.alpha_optim = AlphaOptimCls(
                [self.log_alpha],
                **self.otil_config.get("alpha_optim_kwargs", {}),
            )

        self.actor_lr = self.actor_optim.defaults["lr"]
        self.critic_lr = self.critic_optim.defaults["lr"]
        self.min_lr, self.max_lr = self.otil_config.get("min_lr", 1e-5), self.otil_config.get("max_lr", self.actor_lr)
        # kl scheduler
        self.last_lr = self.actor_lr
        scheduler_kwargs = self.otil_config.get("scheduler_kwargs", {})
        self.scheduler_kwargs = {
            **scheduler_kwargs,
            **{"min_lr": self.min_lr, "max_lr": self.max_lr},
        }
        self.avg_kl = self.scheduler_kwargs.get("kl_threshold", None)

        # --- Loss ---
        cfg = BestOfKConfig(
            T=self.horizon_len,
            K=8,
            eps=0.1,
            sinkhorn_iters=60,
            tau=0.5,
            use_mlp_features=False,
            feature_dim=self.num_obs,
            embed_dim=64,
            use_huber=False,
            huber_delta=1.0,
            action_weight=1.0,
        )
        self.loss_fn = BestOfK(cfg, device=self.device)

        # --- Replay Buffer ---
        assert self.num_actors == self.env.num_envs
        T, B = self.horizon_len, self.num_envs
        self.create_buffers(T, B)

        # --- Episode Metrics ---
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=int, device=self.device)

        self.episode_rewards_hist = []
        self.episode_lengths_hist = []

        tracker_len = 100
        self.episode_rewards_tracker = Tracker(tracker_len)
        self.episode_lengths_tracker = Tracker(tracker_len)
        self.num_episodes = torch.tensor(0, dtype=int)

        # --- Timing ---
        self.timer = Timer()

    def create_buffers(self, T, B):
        self.obs_buf = {k: torch.zeros((T, B) + v, dtype=torch.float32, device=self.device) for k, v in self.obs_space.items()}
        self.action_buf = torch.zeros((T, B, self.action_dim), dtype=torch.float32, device=self.device)
        # for kl divergence computing
        self.mus = torch.zeros((T, B, self.num_actions), dtype=torch.float32, device=self.device)
        self.sigmas = torch.zeros((T, B, self.num_actions), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.next_values = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.target_values = torch.zeros((T, B), dtype=torch.float32, device=self.device)

    def get_actions(self, obs, z=None, sample=True, dist=False):
        # NOTE: obs_rms.normalize(...) occurs elsewhere
        if z is None:
            z = self.actor_encoder(obs)
        if self.actor_detach_z:
            if isinstance(z, dict):
                z = {k: v.detach() for k, v in z.items()}
            else:
                z = z.detach()
        mu, sigma, distr = self.actor(z)
        if sample:
            actions = distr.rsample()
        else:
            actions = mu

        if self.tanh_clamp:
            # clamp actions
            actions = torch.tanh(actions)

        if dist:
            return actions, mu, sigma, distr
        else:
            return actions

    @torch.no_grad()
    def evaluate_policy(self, num_episodes, sample=False, render=False):
        episode_rewards_hist = []
        episode_lengths_hist = []
        episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        episode_lengths = torch.zeros(self.num_envs, dtype=int)

        completed_episodes = {}
        completed_episodes_returns = []
        completed_episodes_lengths = []
        done_ids = torch.zeros(self.num_envs, dtype=int, device=self.device)

        rollouts = {
            env_num: {
                "obs": {k: [] for k in self.obs_space.keys()},
                "next_obs": {k: [] for k in self.obs_space.keys()},
                "actions": [],
                "rew": [],
                "done": [],
            }
            for env_num in range(self.num_envs)
        }
        obs = self.env.reset()
        obs = self._convert_obs(obs)

        episodes = 0
        while episodes < num_episodes:
            for env_num in range(self.num_envs):
                for k, v in obs.items():
                    rollouts[env_num]["obs"][k].append(v[env_num].clone())

            if self.obs_rms is not None:
                obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}

            actions = self.get_actions(obs, sample=sample)
            obs, rew, done, info = self.env.step(actions)
            # if obs_before_reset is available, use it, otherwise use obs
            real_obs = info.get("obs_before_reset", obs)
            if real_obs is None:
                real_obs = obs

            obs = self._convert_obs(obs)
            real_obs = self._convert_obs(real_obs)

            for env_num in range(self.num_envs):
                for k, v in real_obs.items():
                    rollouts[env_num]["next_obs"][k].append(v[env_num].clone())
                rollouts[env_num]["actions"].append(actions[env_num])
                rollouts[env_num]["rew"].append(rew[env_num])
                rollouts[env_num]["done"].append(done[env_num])

            episode_rewards += rew
            episode_lengths += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    done_env_id = int(done_env_id)
                    done_ids[done_env_id] += 1
                    if done_ids[done_env_id] > 2:
                        continue

                    print('rew = {:.2f}, len = {}'.format(episode_rewards[done_env_id].item(), episode_lengths[done_env_id]))
                    episode_rewards_hist.append(episode_rewards[done_env_id].item())
                    episode_lengths_hist.append(episode_lengths[done_env_id].item())
                    episode_rewards[done_env_id] = 0.0
                    episode_lengths[done_env_id] = 0

                    completed_episodes[episodes] = {
                        "obs": {k: torch.stack(v) for k, v in rollouts[done_env_id]["obs"].items()},
                        "act": torch.stack(rollouts[done_env_id]["actions"]),
                        "next_obs": {k: torch.stack(v) for k, v in rollouts[done_env_id]["next_obs"].items()},
                        "rew": torch.stack(rollouts[done_env_id]["rew"]),
                        "done": torch.stack(rollouts[done_env_id]["done"]),
                    }
                    completed_episodes_returns.append(completed_episodes[episodes]["rew"].sum().item())
                    completed_episodes_lengths.append(completed_episodes[episodes]["rew"].shape[0])
                    episodes += 1
                    rollouts[done_env_id] = {
                        "obs": {k: [] for k in self.obs_space.keys()},
                        "next_obs": {k: [] for k in self.obs_space.keys()},
                        "actions": [],
                        "rew": [],
                        "done": [],
                    }
        mean_completed_episode_return = np.mean(completed_episodes_returns)
        mean_completed_episodes_lengths = np.mean(completed_episodes_lengths)
        save_path = os.path.join(self.logdir, "demos")
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            task_name = self.full_cfg.task.name
            env_name = self.full_cfg.task.env.env_name

            print(env_name, task_name)
            torch.save(
                completed_episodes,
                os.path.join(
                    save_path,
                    f"{task_name}_{env_name}_demos{len(completed_episodes)}_epochs{self.epoch}_steps{self.agent_steps}_return{int(mean_completed_episode_return)}_len{int(mean_completed_episodes_lengths)}.pt",
                ),
            )

        return episode_rewards_hist, episode_lengths_hist

    def initialize_env(self):
        try:
            self.env.clear_grad()
        except Exception as e:
            print(e)
            print("Skipping clear_grad")
        self.env.reset()

    def train(self):
        # initializations
        self.initialize_env()

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            if self.max_epochs > 0 and self.epoch >= self.max_epochs:
                break

            # learning rate schedule
            if self.otil_config.lr_schedule == "linear":
                assert self.max_epochs > 0
                actor_lr = (self.min_lr - self.actor_lr) * float(self.epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optim.param_groups:
                    param_group["lr"] = actor_lr
                lr = actor_lr
            elif self.otil_config.lr_schedule == "constant":
                lr = self.actor_lr
            elif self.otil_config.lr_schedule == "kl":
                if self.avg_kl is not None:
                    actor_lr = adaptive_scheduler(self.last_lr, self.avg_kl.item(), **self.scheduler_kwargs)
                    for param_group in self.actor_optim.param_groups:
                        param_group["lr"] = actor_lr
                    self.last_lr = actor_lr
                lr = self.last_lr
            else:
                raise NotImplementedError(self.otil_config.lr_schedule)

            # train actor
            self.timer.start("train/update_actor")
            self.actor_encoder.train()
            self.actor.train()
            self.critic.eval()
            self.critic_target.eval()
            actor_results = self.update_actor()
            self.timer.end("train/update_actor")

            # train critic
            self.timer.start("train/make_critic_dataset")
            with torch.no_grad():
                self.compute_target_values()
                values_results = {
                    "target_values/mean": self.target_values.mean().item(),
                    "target_values/std": self.target_values.std().item(),
                    "target_values/max": self.target_values.max().item(),
                    "target_values/min": self.target_values.min().item(),
                }
                target_values = self.target_values.clone()

            self.encoder.train()
            self.critic.train()
            dataset = CriticDataset(self.critic_batch_size, self.obs_buf, target_values, drop_last=False)
            self.timer.end("train/make_critic_dataset")

            self.timer.start("train/update_critic")
            critic_results = self.update_critic(dataset)
            self.timer.end("train/update_critic")
            self.encoder.eval()
            self.critic.eval()

            if not self.no_target_critic:
                with torch.no_grad():
                    alpha = self.target_critic_alpha
                    soft_update(self.encoder, self.encoder_target, alpha)
                    soft_update(self.critic, self.critic_target, alpha)

            # train metrics
            results = {**actor_results, **critic_results}
            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            if "mu" in results:
                metrics["mu"] = torch.mean(torch.cat(results["mu"]), 0).cpu().numpy()
            if "sigma" in results:
                metrics["sigma"] = torch.mean(torch.cat(results["sigma"]), 0).cpu().numpy()
            metrics.update(values_results)
            metrics.update({"epoch": self.epoch, "lr": lr})
            if self.with_autoent:
                metrics["entropy_alpha"] = self.get_alpha(scalar=True)
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            # timing metrics
            timings_total_names = (
                "train/update_actor",
                "train/make_critic_dataset",
                "train/update_critic",
            )
            timings = self.timer.stats(step=self.agent_steps, total_names=timings_total_names, reset=False)
            timing_metrics = {f"train_timings/{k}": v for k, v in timings.items()}
            metrics.update(timing_metrics)

            # episode metrics
            if len(self.episode_rewards_hist) > 0:
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
                mean_episode_lengths = 0

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            self._checkpoint_save(mean_episode_rewards)

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(
                    f"Epochs: {self.epoch + 1} |",
                    f"Agent Steps: {int(self.agent_steps):,} |",
                    f'SPS: {timings["lastrate"]:.2f} |',  # actually totalrate since we don't reset the timer
                    f'Best: {self.best_stat if self.best_stat is not None else -float("inf"):.2f} |',
                    "Stats:",
                    f"actor_loss {metrics['train_stats/actor_loss']:.2f},",
                    f"ep_rewards {mean_episode_rewards:.2f},",
                    f"ep_lengths {mean_episode_lengths:.2f},",
                    f'grad_norm_before_clip/actor {metrics["train_stats/grad_norm_before_clip/actor"]:.2f},',
                    f'grad_norm_after_clip/actor {metrics["train_stats/grad_norm_after_clip/actor"]:.2f},',
                    f"expert return {self.demos['expert_return']:.2f}",
                    "\b\b |",
                )

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, "final.pth"))

        # save reward/length history
        self.episode_rewards_hist = np.array(self.episode_rewards_hist)
        self.episode_lengths_hist = np.array(self.episode_lengths_hist)
        np.save(
            open(os.path.join(self.logdir, "ep_rewards_hist.npy"), "wb"),
            self.episode_rewards_hist,
        )
        np.save(
            open(os.path.join(self.logdir, "ep_lengths_hist.npy"), "wb"),
            self.episode_lengths_hist,
        )

    def update_actor(self):
        results = collections.defaultdict(list)

        # zero out just in case
        with torch.no_grad():
            self.action_buf.zero_()
            self.mus.zero_()
            self.sigmas.zero_()
            self.rew_buf.zero_()
            self.done_mask.zero_()
            self.next_values.zero_()
            self.target_values.zero_()

        def actor_closure():
            self.actor_optim.zero_grad()
            self.timer.start("train/actor_closure/actor_loss")

            self.timer.start("train/actor_closure/forward_sim")
            actor_loss, info = self.compute_actor_loss()
            self.timer.end("train/actor_closure/forward_sim")

            loss = actor_loss
            self.timer.start("train/actor_closure/backward_sim")
            loss.backward()
            self.timer.end("train/actor_closure/backward_sim")

            with torch.no_grad():
                grad_norm_before_clip = grad_norm(self.actor.parameters())
                if self.otil_config.truncate_grads:
                    if self.otil_config.get("max_grad_value", None) is not None:
                        nn.utils.clip_grad_value_(self.actor.parameters(), self.otil_config.max_grad_value)
                    # elif self.otil_config.get("actor_agc_clip", None) is not None:
                    #     clip_agc_(
                    #         self.actor.parameters(), self.otil_config.actor_agc_clip
                    #     )
                    elif self.otil_config.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.otil_config.max_grad_norm)
                grad_norm_after_clip = grad_norm(self.actor.parameters())

                # sanity check
                if torch.isnan(grad_norm_before_clip) or grad_norm_before_clip > 1e6:
                    print("NaN gradient - skipping update", grad_norm_before_clip)
                    # raise ValueError
                    # raise KeyboardInterrupt
                    self.actor_optim.zero_grad()

            for stat_key in (
                "Lk_min",
                "Lk_mean",
                "pseudo_reward_mean",
                "pseudo_reward_std",
                "imitation_loss",
                "value_term",
                "value_return_mean",
                "value_return_std",
                "bootstrap_value_mean",
                "entropy_mean",
                "entropy_bonus_mean",
            ):
                if stat_key in info:
                    results[stat_key].append(
                        info[stat_key].detach() if torch.is_tensor(info[stat_key]) else torch.tensor(info[stat_key])
                    )
            results["actor_loss"].append(actor_loss.detach())
            results["grad_norm_before_clip/actor"].append(grad_norm_before_clip)
            results["grad_norm_after_clip/actor"].append(grad_norm_after_clip)
            self.timer.end("train/actor_closure/actor_loss")
            return actor_loss

        self.actor_optim.step(actor_closure)

        if self.with_autoent and self.alpha_optim is not None and self._entropy is not None:
            entropy = self._entropy
            alpha = self.get_alpha(detach=False)
            if self.unscale_entropy_alpha:
                if self.offset_by_target_entropy:
                    pass
                if self.scale_by_target_entropy:
                    alpha = alpha * abs(self.target_entropy)
            alpha_loss = (alpha * (entropy - self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            if self.otil_config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.alpha_optim.param_groups[0]["params"], self.otil_config.max_grad_norm)
            self.alpha_optim.step()
            results["entropy_alpha_loss"].append(alpha_loss.detach())

        with torch.no_grad():
            obs = {k: v.view(-1, *v.shape[2:]) for k, v in self.obs_buf.items()}
            _, mu, sigma, distr = self.get_actions(obs, sample=False, dist=True)
            old_mu, old_sigma = self.mus.view(-1, self.num_actions), self.sigmas.view(-1, self.num_actions)

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
            if self.obs_rms is not None:
                obs_rms = deepcopy(self.obs_rms)

        self._entropy = None
        need_entropy = self.track_entropy
        if need_entropy:
            logprob_acc = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            distr_ent_acc = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        if self.with_autoent:
            alpha_scalar = self.get_alpha(scalar=True)
        elif self.entropy_coef is not None:
            alpha_scalar = self.entropy_coef
        else:
            alpha_scalar = None

        # initialize trajectory to cut off gradients between episodes.
        obs = self.env.initialize_trajectory()
        obs = self._convert_obs(obs)

        # collete trajectories and compute actor loss
        obs_window = {k: [] for k in obs.keys()}
        next_values = torch.zeros((self.horizon_len + 1, self.num_envs), dtype=torch.float32, device=self.device)

        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_rms[k].update(v)
            # normalize the current obs
            obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

        for i in range(self.horizon_len):
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_buf[k][i] = v.clone()

            # take env step
            z = self.actor_encoder(obs)
            actions, mu, sigma, distr = self.get_actions(obs, z=z, sample=True, dist=True)

            if need_entropy:
                logprob = distr.log_prob(actions).sum(dim=-1)
                distr_ent = distr.entropy().sum(dim=-1)

            with torch.no_grad():
                self.action_buf[i] = actions.clone()
                self.mus[i, ...] = mu.clone()
                self.sigmas[i, ...] = sigma.clone()
            if need_entropy:
                logprob_acc += logprob
                distr_ent_acc += distr_ent

            obs, rew, done, info = self.env.step(actions)
            # if obs_before_reset is available, use it, otherwise use obs
            real_obs = info.get("obs_before_reset", obs)
            if real_obs is None:
                real_obs = obs

            obs = self._convert_obs(obs)
            real_obs = self._convert_obs(real_obs)

            with torch.no_grad():
                raw_rew = rew.clone()

            # update episode metrics
            with torch.no_grad():
                self.episode_rewards += raw_rew
                self.episode_lengths += 1

            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    for k, v in obs.items():
                        self.obs_rms[k].update(v)
                # normalize the current obs
                obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}
                real_obs = {k: obs_rms[k].normalize(v) for k, v in real_obs.items()}

            with torch.no_grad():
                z_target = self.encoder_target(obs)
                pred_val = self.critic_target(z_target, return_type="min").squeeze(-1)
                next_values[i + 1] = pred_val

            for k, v in real_obs.items():
                obs_window[k].append(v)

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            # collect episode metrics
            with torch.no_grad():
                if len(done_env_ids) > 0:
                    done_env_ids = done_env_ids.detach().cpu()
                    terminal_obs = info.get("obs_before_reset", None)
                    if terminal_obs is None:
                        terminal_obs = obs
                    terminal_obs = self._convert_obs(terminal_obs)
                    if self.obs_rms is not None:
                        terminal_obs = {k: obs_rms[k].normalize(v) for k, v in terminal_obs.items()}
                    self.episode_rewards_tracker.update(self.episode_rewards[done_env_ids])
                    self.episode_lengths_tracker.update(self.episode_lengths[done_env_ids])
                    self.num_episodes += len(done_env_ids)
                    for done_env_id in done_env_ids:
                        if self.episode_rewards[done_env_id] > 1e6 or self.episode_rewards[done_env_id] < -1e6:
                            print("ep_rewards error")
                            raise ValueError
                        self.episode_rewards_hist.append(self.episode_rewards[done_env_id].item())
                        self.episode_lengths_hist.append(self.episode_lengths[done_env_id].item())
                        real_obs_term = {k: v[done_env_id : done_env_id + 1] for k, v in terminal_obs.items()}
                        nan_obs = False
                        for obs_key, obs_val in real_obs_term.items():
                            if torch.isnan(obs_val).any() or torch.isinf(obs_val).any() or (torch.abs(obs_val) > 1e6).any():
                                nan_obs = True
                                break
                        if nan_obs or self.episode_lengths[done_env_id] < self.max_episode_length:
                            next_values[i + 1, done_env_id] = 0.0
                        else:
                            real_z_target = self.encoder_target(real_obs_term)
                            real_next_values = self.critic_target(real_z_target, return_type="min").squeeze(-1)
                            next_values[i + 1, done_env_id] = real_next_values
                        self.episode_rewards[done_env_id] = 0.0
                        self.episode_lengths[done_env_id] = 0

            with torch.no_grad():
                if i < self.horizon_len - 1:
                    self.done_mask[i] = done.to(dtype=torch.float32)
                else:
                    self.done_mask[i] = torch.ones_like(done, dtype=torch.float32)
                self.next_values[i] = next_values[i + 1].clone()

        entropy_bonus = None
        entropy_bonus_detached = None
        entropy_mean = None
        entropy_bonus_mean = None
        raw_entropy = None
        if need_entropy:
            raw_entropy = distr_ent_acc if self.use_distr_ent else -logprob_acc
            self._entropy = raw_entropy.detach().clone()
            entropy_mean = raw_entropy.mean().detach()
            if alpha_scalar is not None:
                entropy_processed = raw_entropy.clone()
                if self.offset_by_target_entropy:
                    entropy_processed = (entropy_processed + abs(self.target_entropy)) * 0.5
                if self.scale_by_target_entropy:
                    entropy_processed = entropy_processed * (1.0 / abs(self.target_entropy))
                alpha_tensor = entropy_processed.new_tensor(alpha_scalar)
                entropy_bonus = alpha_tensor * entropy_processed
                entropy_bonus_detached = entropy_bonus.detach()
                entropy_bonus_mean = entropy_bonus_detached.mean()

        obs_window = {k: torch.stack(v, dim=1) for k, v in obs_window.items()}
        final_window_obs = {k: v[:, -1] for k, v in obs_window.items()}
        with torch.no_grad():
            if self.obs_rms is not None:
                obs_exp = {k: obs_rms[k].normalize(v) for k, v in self.demos["obs"].items()}

        obs_z = self.actor_encoder(obs_window)
        exp_z = self.actor_encoder(obs_exp).detach()
        loss, info = self.loss_fn(obs_z, exp_z, sim_is_window=False)

        if entropy_mean is not None:
            info["entropy_mean"] = entropy_mean
        if entropy_bonus_mean is not None:
            info["entropy_bonus_mean"] = entropy_bonus_mean

        pseudo_rewards = self._build_pseudo_rewards(info["Lk"])
        if entropy_bonus_detached is not None and self.entropy_in_return:
            pseudo_rewards = pseudo_rewards + entropy_bonus_detached

        pseudo_rewards_detached = pseudo_rewards.detach()
        target_rewards = pseudo_rewards_detached
        if entropy_bonus_detached is not None and self.entropy_in_targets and not self.entropy_in_return:
            target_rewards = target_rewards + entropy_bonus_detached
        with torch.no_grad():
            self.rew_buf[-1] = target_rewards
        info["pseudo_reward_mean"] = pseudo_rewards_detached.mean()
        info["pseudo_reward_std"] = pseudo_rewards_detached.std(unbiased=False)

        info["imitation_loss"] = loss.detach()

        value_term = None
        if self.actor_value_coef != 0.0:
            value_term, value_stats = self._compute_actor_value_term(final_window_obs, pseudo_rewards_detached)
            info.update(value_stats)
            info["value_term"] = value_term.detach()
        else:
            value_term = loss.new_tensor(0.0)

        actor_loss = self.actor_loss_coef * loss + self.actor_value_coef * value_term
        if entropy_bonus is not None and not self.entropy_in_return:
            actor_loss = actor_loss - entropy_bonus.mean()

        self.agent_steps += self.horizon_len * self.num_envs
        return actor_loss, info

    def _build_pseudo_rewards(self, Lk: torch.Tensor) -> torch.Tensor:
        if self.critic_reward_reduction == "min":
            rewards = -torch.log(1 - torch.exp(-Lk.min(dim=1).values) + 1e-10)
        elif self.critic_reward_reduction == "mean":
            rewards = -Lk.mean(dim=1)
        elif self.critic_reward_reduction == "softmin":
            rewards = -(-self.critic_reward_tau * torch.logsumexp(-Lk / self.critic_reward_tau, dim=1))
        else:
            raise NotImplementedError(self.critic_reward_reduction)

        if self.critic_reward_normalize:
            rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)

        rewards = self.reward_shaper(rewards)
        rewards = rewards * self.critic_reward_scale
        return rewards

    def _compute_actor_value_term(self, final_obs, pseudo_rewards):
        if self.share_encoder:
            z_value = self.actor_encoder(final_obs)
        else:
            z_value = self.encoder_target(final_obs)

        value_pred = self.critic_target(z_value, return_type=self.actor_value_return_type).squeeze(-1)
        value_returns = pseudo_rewards + self.gamma * value_pred
        value_term = -value_returns.mean()
        stats = {
            "value_return_mean": value_returns.mean().detach(),
            "value_return_std": value_returns.std(unbiased=False).detach(),
            "bootstrap_value_mean": value_pred.mean().detach(),
        }
        return value_term, stats

    def compute_target_values(self):
        if self.critic_method == "one-step":
            self.target_values = self.rew_buf + self.gamma * (1.0 - self.done_mask) * self.next_values
        elif self.critic_method == "td-lambda":
            lam_acc = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for i in reversed(range(self.horizon_len)):
                mask = self.done_mask[i]
                bootstrap = (1.0 - mask) * self.next_values[i]
                lam_acc = self.rew_buf[i] + self.gamma * ((1.0 - mask) * (self.lam * lam_acc) + bootstrap)
                self.target_values[i] = lam_acc
        else:
            raise NotImplementedError(self.critic_method)

    def update_critic(self, dataset):
        results = collections.defaultdict(list)
        if self.critic_iterations == 0:
            return results

        for _ in range(self.critic_iterations):
            total_critic_loss = 0.0
            grad_norms_before = []
            grad_norms_after = []
            for batch_idx in range(len(dataset)):
                batch_obs, batch_targets = dataset[batch_idx]
                self.critic_optim.zero_grad()
                critic_loss = self.compute_critic_loss(batch_obs, batch_targets)
                critic_loss.backward()
                print(f"Critic loss: {critic_loss.item():.6f}")

                grad_before = grad_norm(self.critic.parameters())
                grad_norms_before.append(grad_before)
                if self.otil_config.truncate_grads:
                    if self.otil_config.get("max_grad_value", None) is not None:
                        nn.utils.clip_grad_value_(self.critic.parameters(), self.otil_config.max_grad_value)
                    elif self.otil_config.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.otil_config.max_grad_norm)
                grad_after = grad_norm(self.critic.parameters())
                grad_norms_after.append(grad_after)

                self.critic_optim.step()
                total_critic_loss += critic_loss.detach()

            value_loss = total_critic_loss / len(dataset)
            results["value_loss"].append(value_loss)
            results["grad_norm_before_clip/critic"].append(torch.mean(torch.stack(grad_norms_before)))
            results["grad_norm_after_clip/critic"].append(torch.mean(torch.stack(grad_norms_after)))

        return results

    def compute_critic_loss(self, obs, target_v):
        z = self.encoder(obs)
        pred_vs = self.critic(z, return_type="all")
        target_v = target_v.to(self.device)
        losses = []
        for pred_v in pred_vs:
            print(pred_v, target_v)
            input("debug")
            losses.append(F.mse_loss(pred_v.squeeze(-1), target_v, reduction="mean"))
        critic_loss = torch.stack(losses).mean()
        return critic_loss

    def get_alpha(self, detach=True, scalar=False):
        if self.otil_config.get("alpha", None) is None:
            if self.log_alpha is None:
                raise ValueError("log_alpha is not initialized")
            alpha = self.log_alpha.exp()
            if detach:
                alpha = alpha.detach()
            if scalar:
                alpha = alpha.item()
        else:
            alpha = self.otil_config.alpha
        return alpha

    def eval(self):
        self.set_eval()

        episode_rewards, episode_lengths = self.evaluate_policy(num_episodes=self.num_actors * 2, sample=True)

        metrics = {
            "eval_scores/num_episodes": len(episode_rewards),
            "eval_scores/episode_rewards": np.mean(np.array(episode_rewards)),
            "eval_scores/episode_lengths": np.mean(np.array(episode_lengths)),
        }
        print(metrics)

        self.writer.add(self.agent_steps, metrics)
        self.writer.write()

        scores = {
            "epoch": self.epoch,
            "mini_epoch": self.mini_epoch,
            "agent_steps": self.agent_steps,
            "eval_scores/num_episodes": len(episode_rewards),
            "eval_scores/episode_rewards": episode_rewards,
            "eval_scores/episode_lengths": episode_lengths,
        }
        json.dump(scores, open(os.path.join(self.logdir, "scores.json"), "w"), indent=4)

    def set_train(self):
        pass

    def set_eval(self):
        self.actor_encoder.eval()
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        if not self.no_target_critic:
            self.encoder_target.eval()
            self.critic_target.eval()

    def save(self, f):
        ckpt = {
            "epoch": self.epoch,
            "mini_epoch": self.mini_epoch,
            "agent_steps": self.agent_steps,
            "obs_rms": self.obs_rms.state_dict() if self.normalize_input else None,
            "actor_encoder": (self.actor_encoder.state_dict() if not self.share_encoder else None),
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "encoder_target": (self.encoder_target.state_dict() if not self.no_target_critic else None),
            "critic_target": (self.critic_target.state_dict() if not self.no_target_critic else None),
            "log_alpha": (
                self.log_alpha.data
                if (self.with_autoent and self.log_alpha is not None and self.otil_config.get("alpha", None) is None)
                else None
            ),
        }
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=""):
        all_ckpt_keys = ("epoch", "mini_epoch", "agent_steps")
        all_ckpt_keys += (
            "obs_rms",
            "actor_encoder",
            "encoder",
            "actor",
            "critic",
            "encoder_target",
            "critic_target",
            "log_alpha",
        )
        ckpt = torch.load(f, map_location=self.device)
        for k in all_ckpt_keys:
            if not re.match(ckpt_keys, k):
                print(f"Warning: ckpt skipped loading `{k}`")
                continue
            if k == "obs_rms" and (not self.normalize_input):
                continue
            if k == "actor_encoder" and (self.share_encoder):
                continue
            if k in ("encoder_target", "critic_target") and self.no_target_critic:
                continue
            if k == "log_alpha":
                if (
                    self.with_autoent
                    and self.log_alpha is not None
                    and self.otil_config.get("alpha", None) is None
                    and ckpt[k] is not None
                ):
                    self.log_alpha.data = ckpt[k]
                continue

            if hasattr(getattr(self, k), "load_state_dict"):
                getattr(self, k).load_state_dict(ckpt[k])
            else:
                setattr(self, k, ckpt[k])
