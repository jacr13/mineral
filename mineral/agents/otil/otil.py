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
from ...common.timer import Timer
from ...common.tracker import Tracker
from ..agent import Agent
from . import models
from .utils import grad_norm, adaptive_scheduler, policy_kl
from .best_of_k import BestOfKConfig, BestOfKSoftminOT


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
        self.tanh_clamp = self.network_config.get(
            "tanh_clamp", False
        )  # on actions, if not done in actor dist
        self.actor_detach_z = self.otil_config.get("actor_detach_z", False)

        self.horizon_len = self.otil_config.horizon_len
        self.max_epochs = self.otil_config.get(
            "max_epochs", 0
        )  # set to 0 to disable and track by max_agent_steps instead

        demos_path = self.otil_config.demos_path
        assert os.path.exists(
            demos_path
        ), f"OTIL demos path: {demos_path} does not exist"
        self.demos = torch.load(demos_path, map_location=self.device)

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
        self.loss_fn = BestOfKSoftminOT(cfg, device=self.device)

        # --- Normalizers ---
        if self.tanh_clamp:  # legacy
            # unbiased=False -> correction=0
            # https://github.com/NVlabs/DiffRL/blob/a4c0dd1696d3c3b885ce85a3cb64370b580cb913/utils/running_mean_std.py#L34
            rms_config = dict(
                eps=1e-5, correction=0, initial_count=1e-4, dtype=torch.float32
            )
        else:
            rms_config = dict(eps=1e-5, initial_count=1, dtype=torch.float64)
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
            self.encoder = EncoderCls(
                self.obs_space, encoder_kwargs, weight_init_fn=models.weight_init_
            )
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
        self.actor = ActorCls(
            obs_dim, self.action_dim, **self.network_config.get("actor_kwargs", {})
        )
        self.actor.to(self.device)
        print("Actor:", self.actor)

        # --- Optim ---
        OptimCls = getattr(torch.optim, self.otil_config.optim_type)

        if self.otil_config.get("actor_detach_encoder", False):
            actor_optim_params = self.actor.parameters()
        else:
            actor_optim_params = itertools.chain(
                self.actor_encoder.parameters(), self.actor.parameters()
            )
        self.actor_optim = OptimCls(
            actor_optim_params,
            **self.otil_config.get("actor_optim_kwargs", {}),
        )
        print("Actor Optim:", self.actor_optim)

        self.actor_lr = self.actor_optim.defaults["lr"]
        self.min_lr, self.max_lr = self.otil_config.get(
            "min_lr", 1e-5
        ), self.otil_config.get("max_lr", self.actor_lr)
        # kl scheduler
        self.last_lr = self.actor_lr
        scheduler_kwargs = self.otil_config.get("scheduler_kwargs", {})
        self.scheduler_kwargs = {
            **scheduler_kwargs,
            **dict(min_lr=self.min_lr, max_lr=self.max_lr),
        }
        self.avg_kl = self.scheduler_kwargs.get("kl_threshold", None)

        # --- Replay Buffer ---
        assert self.num_actors == self.env.num_envs
        T, B = self.horizon_len, self.num_envs
        self.create_buffers(T, B)

        # --- Episode Metrics ---
        self.episode_rewards = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
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

        # for kl divergence computing
        self.mus = torch.zeros(
            (T, B, self.num_actions), dtype=torch.float32, device=self.device
        )
        self.sigmas = torch.zeros(
            (T, B, self.num_actions), dtype=torch.float32, device=self.device
        )
        # TODO complete
        pass

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
        episode_rewards = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        episode_lengths = torch.zeros(self.num_envs, dtype=int)

        completed_episodes = {}
        completed_episodes_returns = []
        completed_episodes_lengths = []

        episode_obs = []
        episode_act = []
        episode_next_obs = []
        episode_rew = []
        episode_done = []
        dones_ids = []

        obs = self.env.reset()
        obs = self._convert_obs(obs)

        episodes = 0
        while episodes < num_episodes:
            episode_obs.append(obs["obs"].clone())
            if self.obs_rms is not None:
                obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}

            actions = self.get_actions(obs, sample=sample)
            obs, rew, done, info = self.env.step(actions)
            obs = self._convert_obs(obs)

            real_obs = info["obs_before_reset"]
            episode_act.append(actions)
            episode_next_obs.append(real_obs)
            episode_rew.append(rew)
            episode_done.append(done)

            episode_rewards += rew
            episode_lengths += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                ep_obs = torch.stack(episode_obs).transpose(0, 1)
                ep_act = torch.stack(episode_act).transpose(0, 1)
                ep_next_obs = torch.stack(episode_next_obs).transpose(0, 1)
                ep_rew = torch.stack(episode_rew).transpose(0, 1)
                ep_done = torch.stack(episode_done).transpose(0, 1)

                for done_env_id in done_env_ids:
                    print(
                        "rew = {:.2f}, len = {}".format(
                            episode_rewards[done_env_id].item(),
                            episode_lengths[done_env_id],
                        )
                    )
                    episode_rewards_hist.append(episode_rewards[done_env_id].item())
                    episode_lengths_hist.append(episode_lengths[done_env_id].item())
                    episode_rewards[done_env_id] = 0.0
                    episode_lengths[done_env_id] = 0
                    episodes += 1

                    completed_episodes[episodes] = {
                        "obs": ep_obs[done_env_id],
                        "act": ep_act[done_env_id],
                        "next_obs": ep_next_obs[done_env_id],
                        "rew": ep_rew[done_env_id],
                        "done": ep_done[done_env_id],
                    }
                    completed_episodes_returns.append(ep_rew[done_env_id].sum().item())
                    completed_episodes_lengths.append(ep_rew[done_env_id].shape[0])

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

        return (
            episode_rewards_hist,
            episode_lengths_hist,
        )

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
                actor_lr = (self.min_lr - self.actor_lr) * float(
                    self.epoch / self.max_epochs
                ) + self.actor_lr
                for param_group in self.actor_optim.param_groups:
                    param_group["lr"] = actor_lr
                lr = actor_lr
            elif self.otil_config.lr_schedule == "constant":
                lr = self.actor_lr
            elif self.otil_config.lr_schedule == "kl":
                if self.avg_kl is not None:
                    actor_lr = adaptive_scheduler(
                        self.last_lr, self.avg_kl.item(), **self.scheduler_kwargs
                    )
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
            actor_results = self.update_actor()
            self.timer.end("train/update_actor")

            # train metrics
            results = {**actor_results}
            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics.update(
                {
                    k: torch.mean(torch.cat(results[k]), 0).cpu().numpy()
                    for k in ["mu", "sigma"]
                }
            )  # distr
            metrics.update({"epoch": self.epoch, "lr": lr})
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            # timing metrics
            timings_total_names = (
                "train/update_actor",
                "train/make_critic_dataset",
                "train/update_critic",
            )
            timings = self.timer.stats(
                step=self.agent_steps, total_names=timings_total_names, reset=False
            )
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
                    f"Stats:",
                    f"actor_loss {metrics['train_stats/actor_loss']:.2f},",
                    f"ep_rewards {mean_episode_rewards:.2f},",
                    f"ep_lengths {mean_episode_lengths:.2f},",
                    f'grad_norm_before_clip/actor {metrics["train_stats/grad_norm_before_clip/actor"]:.2f},',
                    f'grad_norm_after_clip/actor {metrics["train_stats/grad_norm_after_clip/actor"]:.2f},',
                    f"\b\b |",
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
                        nn.utils.clip_grad_value_(
                            self.actor.parameters(), self.otil_config.max_grad_value
                        )
                    # elif self.otil_config.get("actor_agc_clip", None) is not None:
                    #     clip_agc_(
                    #         self.actor.parameters(), self.otil_config.actor_agc_clip
                    #     )
                    elif self.otil_config.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.actor.parameters(), self.otil_config.max_grad_norm
                        )
                grad_norm_after_clip = grad_norm(self.actor.parameters())

                # sanity check
                if torch.isnan(grad_norm_before_clip) or grad_norm_before_clip > 1e6:
                    print("NaN gradient", grad_norm_before_clip)
                    # raise ValueError
                    raise KeyboardInterrupt
            results["actor_loss"].append(actor_loss.detach())
            results["grad_norm_before_clip/actor"].append(grad_norm_before_clip)
            results["grad_norm_after_clip/actor"].append(grad_norm_after_clip)
            self.timer.end("train/actor_closure/actor_loss")
            return actor_loss

        self.actor_optim.step(actor_closure)

        with torch.no_grad():
            obs = {k: v.view(-1, *v.shape[2:]) for k, v in self.obs_buf.items()}
            _, mu, sigma, distr = self.get_actions(obs, sample=False, dist=True)
            old_mu, old_sigma = self.mus.view(-1, self.num_actions), self.sigmas.view(
                -1, self.num_actions
            )

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

        # initialize trajectory to cut off gradients between episodes.
        obs = self.env.initialize_trajectory()
        obs = self._convert_obs(obs)

        # collete trajectories and compute actor loss
        obs_list = []  # TODO: fix convert to tensor.zeros

        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_rms[k].update(v)
            # normalize the current obs
            obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

        for i in range(self.horizon_len):
            # take env step
            z = self.actor_encoder(obs)
            actions, mu, sigma, distr = self.get_actions(
                obs, z=z, sample=True, dist=True
            )

            with torch.no_grad():
                self.action_buf[i] = actions.clone()
                self.mus[i, ...] = mu.clone()
                self.sigmas[i, ...] = sigma.clone()

            obs, rew, done, extra_info = self.env.step(actions)
            obs = self._convert_obs(obs)

            real_obs = extra_info["obs_before_reset"]
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

            obs_list.append(real_obs["obs"])

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            # collect episode metrics
            with torch.no_grad():
                if len(done_env_ids) > 0:
                    done_env_ids = done_env_ids.detach().cpu()
                    self.episode_rewards_tracker.update(
                        self.episode_rewards[done_env_ids]
                    )
                    self.episode_lengths_tracker.update(
                        self.episode_lengths[done_env_ids]
                    )
                    self.num_episodes += len(done_env_ids)
                    for done_env_id in done_env_ids:
                        if (
                            self.episode_rewards[done_env_id] > 1e6
                            or self.episode_rewards[done_env_id] < -1e6
                        ):
                            print("ep_rewards error")
                            raise ValueError
                        self.episode_rewards_hist.append(
                            self.episode_rewards[done_env_id].item()
                        )
                        self.episode_lengths_hist.append(
                            self.episode_lengths[done_env_id].item()
                        )
                        self.episode_rewards[done_env_id] = 0.0
                        self.episode_lengths[done_env_id] = 0

        obs_list = torch.stack(obs_list, dim=1).to(self.device)
        loss, info = self.loss_fn(obs_list, self.demos["obs"], sim_is_window=False)

        self.agent_steps += self.horizon_len * self.num_envs
        return loss, info

    def eval(self):
        self.set_eval()

        episode_rewards, episode_lengths = self.evaluate_policy(
            num_episodes=self.num_actors * 2, sample=True
        )

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

    def save(self, f):
        ckpt = {
            "epoch": self.epoch,
            "mini_epoch": self.mini_epoch,
            "agent_steps": self.agent_steps,
            "obs_rms": self.obs_rms.state_dict() if self.normalize_input else None,
            "actor_encoder": (
                self.actor_encoder.state_dict() if not self.share_encoder else None
            ),
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
        }
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=""):
        all_ckpt_keys = ("epoch", "mini_epoch", "agent_steps")
        all_ckpt_keys += (
            "obs_rms",
            "actor_encoder",
            "encoder",
            "actor",
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

            if hasattr(getattr(self, k), "load_state_dict"):
                getattr(self, k).load_state_dict(ckpt[k])
            else:
                setattr(self, k, ckpt[k])
