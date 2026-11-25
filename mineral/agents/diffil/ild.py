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


def _chunks(obs, size):
    n = next(iter(obs.values())).shape[0]
    for c in range(0, n, size):
        yield {k: v[c : c + size] for k, v in obs.items()}


class ILD(SHAC):
    """Differentiable ILD that optimizes a Chamfer imitation loss directly."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        self.ild_config = full_cfg.agent.get("ild", {})

        demos_config = self.ild_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)

        self.loss_fn = ChamferImitationLoss(alpha=float(self.ild_config.get("alpha", 1.0)))

        # expert state set: only states up to each demo's first done, finite (padding after an early end is masked)
        done = self.demos["done"].bool()
        N, T = done.shape
        t = torch.arange(T, device=self.device)[None].expand(N, T)
        first_done = torch.where(done.any(1), done.float().argmax(1), torch.full((N,), T - 1, device=self.device))
        valid = t <= first_done[:, None]
        for v in self.demos["obs"].values():
            valid &= torch.isfinite(v.reshape(N, T, -1)).all(-1)
        self.expert_obs = {k: v[valid] for k, v in self.demos["obs"].items()}  # (M, ...)
        from ..il.common import state_dims

        self.state_d = state_dims(full_cfg, self.obs_space, self.ild_config.get("state_obs", False))
        self.demo_valid = valid
        # paper mode: the Chamfer loss compares simulator states (the paper uses Brax's body positions, rotations and
        # velocities), i.e. the state part of the DFlex observation (without the previous action); the policy reads the
        # full observation
        self.feat_d = state_dims(full_cfg, self.obs_space, bool(self.ild_config.get("loss_state", True)))
        self.full_cfg = full_cfg
        print(f"[ILD] expert state set: {int(valid.sum())} states from {N} demos")
        if str(self.ild_config.get("mode", "paper")) == "paper":
            # built here (not in train_paper) so that a resume checkpoint can restore its state before training
            params = list(self.actor_encoder.parameters()) + list(self.actor.parameters())
            self.paper_optim = torch.optim.Adam(params, lr=float(self.ild_config.get("lr", 1e-3)))

    def train(self):
        if str(self.ild_config.get("mode", "paper")) == "paper":
            return self.train_paper()
        return self.train_window()

    def train_window(self):
        """Earlier variant: SHAC-style windows of horizon_len steps, Chamfer against the pooled expert states."""
        if not self._resumed:  # a resumed run has its env state from the resume checkpoint
            self.initialize_env()

        self.protocol_begin()

        while self.agent_steps < self.max_agent_steps:
            self.protocol_step()
            self.epoch += 1
            if self.max_epochs > 0 and self.epoch >= self.max_epochs:
                print("Reached max epochs. Exiting cleanly.")
                break
            if self.job_clock is not None:
                _, safe_stop = self.job_clock.step(check_safe_stop=True)
                if safe_stop:
                    print("Not enough time left for another step. Exiting cleanly.")
                    self.protocol_clock_stop()
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

        self.protocol_end(self.max_agent_steps)

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

        obs_window = {k: torch.stack(v, dim=1) for k, v in obs_window.items()}  # (B, H, ...)
        if obs_rms is not None:
            with torch.no_grad():
                obs_exp = {k: obs_rms[k].normalize(v) for k, v in self.expert_obs.items()}
        else:
            obs_exp = self.expert_obs

        # Chamfer distance in a FIXED feature space: the normalised flat observation, or (dict observations) a
        # frozen copy of the actor encoder; a trainable encoder would let the loss collapse the features
        if "obs" in obs_window and len(obs_window) == 1:  # state part (DFlex obs end with the previous action)
            d = self.state_d or obs_window["obs"].shape[-1]
            obs_z, exp_z = obs_window["obs"][..., :d], obs_exp["obs"][..., :d].unsqueeze(0)
        else:
            frozen = deepcopy(self.actor_encoder).requires_grad_(False)
            B, H = next(iter(obs_window.values())).shape[:2]
            obs_z = frozen({k: v.flatten(0, 1) for k, v in obs_window.items()})['z'].view(B, H, -1)
            with torch.no_grad():
                exp_z = torch.cat([frozen(tree)['z'] for tree in _chunks(obs_exp, 4096)]).unsqueeze(0)
        actor_loss, info = self.loss_fn(obs_z, exp_z)
        info["imitation_loss"] = actor_loss.detach()
        info["expert_return"] = self.demos["expert_return"]

        self.agent_steps += self.horizon_len * self.num_envs
        return actor_loss, info

    # ------------------------------------------------------------------------------------------------------------
    # ILD as in the paper (Chen et al., CVPR 2023, github.com/sail-sg/ILD, policy/brax_task/train_multi_traj.py)
    # ------------------------------------------------------------------------------------------------------------
    # Each iteration rolls out every env for a whole episode from a reset, with gradients cut every trunc_len steps
    # and each step's backward gradient (actions, observations) rescaled to norm <= step_grad_norm. The loss is a
    # per-trajectory Chamfer distance to each demonstration on normalised simulator states (both directions,
    # RMS distance per state pair), the best demonstration kept per env. Adam, lr linearly from lr to min_lr, global
    # gradient clip max_grad_norm. The paper also applies tanh to the final loss; it is left out (ild.loss_tanh: false),
    # since it saturates and stops the gradient. The paper first clones the expert's actions for 1000 iterations;
    # observation-only demos have no actions, so bootstrap: stage1 clones MENTOR's Stage I recovered actions instead
    # (bootstrap: none = no cloning).
    #
    # Memory: a 1000-step DFlex episode does not fit on the tape. Because the gradient is cut every trunc_len steps
    # and, once the nearest-neighbour assignments are fixed, the loss is a sum of per-state terms, the gradient is
    # computed exactly in two passes: pass A rolls out without gradients and fixes the assignments; pass B replays the
    # same rollout (same initial state, RNG state and action noise) segment by segment and back-propagates each
    # segment's share of the loss. Simulator steps per iteration: 2 x episode (forward) + 1 x episode (backward).

    def _feat(self, obs_dict):
        x = obs_dict["obs"]
        return x[..., : self.feat_d] if self.feat_d else x

    def _policy_actions(self, obs_rms, obs, eps):
        o = {k: obs_rms[k].normalize(v) for k, v in obs.items()} if obs_rms is not None else obs
        mu, sigma, _ = self.actor(self.actor_encoder(o))
        a = mu + sigma * eps
        return torch.tanh(a) if self.tanh_clamp else a

    def _stage1_actions(self):
        """MENTOR's Stage I (Levenberg-Marquardt through one simulator step) on every valid demo transition."""
        from .mentor import lm_invert
        from .mentor_sim import OneStep, SimCounter, get_spec

        icfg = dict(self.ild_config.get("bootstrap_invert", {}))
        spec = get_spec(self.full_cfg.task.env.env_name)
        S = self.demos["obs"]["obs"][..., : spec.obs_dim]
        S1 = self.demos["next_obs"]["obs"][..., : spec.obs_dim]
        valid = self.demo_valid
        s, s1 = S[valid], S1[valid]
        w = 1.0 / (s1 - s).std(0).clamp_min(1e-8)
        batch = int(icfg.get("batch", 2000))
        counter = SimCounter()
        step = OneStep(spec, batch, self.device, counter)
        acts = []
        for c0 in range(0, s.shape[0], batch):
            x, y = s[c0 : c0 + batch], s1[c0 : c0 + batch]
            k = x.shape[0]
            if k < batch:
                x, y = torch.cat([x, x[:1].expand(batch - k, -1)]), torch.cat([y, y[:1].expand(batch - k, -1)])
            a, _ = lm_invert(
                step, x, y, w, iters=int(icfg.get("iters", 20)), starts=1, reg=float(icfg.get("reg", 0.0)),
                param=str(icfg.get("param", "box")), jacobian=str(icfg.get("jacobian", "broyden")),
                refresh=int(icfg.get("refresh", 0)), refresh_jacobian=str(icfg.get("refresh_jacobian", "autodiff")),
                h=float(icfg.get("fd_step", 1e-2)),
            )  # fmt: skip
            acts.append(a[:k].detach())
        del step
        print(f"[ILD] Stage I: {s.shape[0]} transitions, {counter.total} simulator steps")
        return torch.cat(acts), counter.total

    def _bootstrap(self, lr):
        mode = str(self.ild_config.get("bootstrap", "none"))
        if mode == "none":
            return 0
        if mode != "stage1":
            raise ValueError(f"ild.bootstrap must be none or stage1, not {mode!r}")
        a_hat, sim_steps = self._stage1_actions()
        obs = {k: v[self.demo_valid] for k, v in self.demos["obs"].items()}
        if self.obs_rms is not None:
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_rms[k].update(v)
        params = list(self.actor_encoder.parameters()) + list(self.actor.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(int(self.ild_config.get("bootstrap_iters", 1000))):
            eps = torch.randn_like(a_hat)
            loss = ((self._policy_actions(self.obs_rms, obs, eps) - a_hat) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, float(self.ild_config.get("max_grad_norm", 0.3)))
            opt.step()
            if it % 200 == 0:
                print(f"[ILD] bootstrap {it}: BC loss {loss.item():.4f}")
        return sim_steps

    def _chamfer_assign(self, X, Y, Ymask, df):
        """X (B, T, D) rollout features, Y (N, L, D) demo features, Ymask (N, L) valid demo states.

        Per env b and demo n: deviation = mean_t min_l d(x_t, y_l), coverage = mean_l min_t d(y_l, x_t), with
        d = RMS difference; total = df * deviation + coverage; keep the best demo n*. Returns the loss and the
        assignments (n*, nearest demo state per rollout state, nearest rollout state per demo state).
        """
        B, T, D = X.shape
        N, L, _ = Y.shape
        best_n = torch.zeros(B, dtype=torch.long, device=X.device)
        nn_l = torch.zeros(B, T, dtype=torch.long, device=X.device)
        nn_t = torch.zeros(B, L, dtype=torch.long, device=X.device)
        total_best = torch.zeros(B, device=X.device)
        n_valid = Ymask.sum(1).clamp_min(1)
        for b in range(B):
            d = torch.cdist(X[b][None].expand(N, T, D), Y) / D**0.5  # (N, T, L)
            dev_min, dev_arg = d.masked_fill(~Ymask[:, None, :], float("inf")).min(2)  # (N, T)
            cov_min, cov_arg = d.min(1)  # (N, L)
            total = df * dev_min.mean(1) + (cov_min * Ymask).sum(1) / n_valid
            n = total.argmin()
            best_n[b], total_best[b] = n, total[n]
            nn_l[b], nn_t[b] = dev_arg[n], cov_arg[n]
        return total_best.mean(), best_n, nn_l, nn_t

    def train_paper(self):
        cfg = self.ild_config
        B = self.num_envs
        T = int(cfg.get("episode_len", 0)) or int(self.env.episode_length)
        trunc = int(cfg.get("trunc_len", 10))
        df = float(cfg.get("deviation_factor", 1.0))
        lr0, lr1 = float(cfg.get("lr", 1e-3)), float(cfg.get("min_lr", 1e-5))
        clip = float(cfg.get("max_grad_norm", 0.3))
        step_clip = float(cfg.get("step_grad_norm", 1.0))
        # count_all_sim_steps: each iteration simulates 2 * B * T steps (a rollout without gradient for the targets,
        # then the rollout that is differentiated); count both, as every other method counts every simulated step,
        # and round the iteration count up so the run reaches its budget. False keeps the earlier count (B * T per
        # iteration, budget rounded down).
        count_all = bool(cfg.get("count_all_sim_steps", False))
        per_it = (2 if count_all else 1) * B * T
        if count_all:
            iters = max(1, -(-int(self.max_agent_steps) // per_it))
        else:
            iters = max(1, int(self.max_agent_steps) // per_it)
        print(f"[ILD] paper mode: {B} envs x {T} steps, truncation {trunc}, {iters} iterations")

        # demo trajectories (first T states), features and validity
        Yraw = self._feat(self.demos["obs"])[:, :T]
        Ymask = self.demo_valid[:, :T]
        Yraw = torch.where(Ymask[..., None], Yraw, torch.zeros_like(Yraw))
        if not self._resumed:  # a resumed run has these (and the iteration to run next) from the checkpoint
            # running state normaliser of the features: mean, variance, count
            self.paper_feat = [torch.zeros(Yraw.shape[-1], device=self.device),
                               torch.ones(Yraw.shape[-1], device=self.device), 0.0]
            self.sim_steps = self._bootstrap(lr0)
            self.paper_next = 0
        params = list(self.actor_encoder.parameters()) + list(self.actor.parameters())
        optim = self.paper_optim

        def step_norm(g):
            n = g.norm()
            return torch.where(n > step_clip, g * (step_clip / n), g)

        self.protocol_begin()
        for it in range(self.paper_next, iters):
            self.paper_next = it
            self.protocol_step()
            self.epoch = it + 1
            if self.job_clock is not None:
                _, safe_stop = self.job_clock.step(check_safe_stop=True)
                if safe_stop:
                    print("Not enough time left for another iteration. Exiting cleanly.")
                    self.protocol_clock_stop()
                    break
            lr = lr0 + (lr1 - lr0) * it / max(1, iters)
            for g in optim.param_groups:
                g["lr"] = lr
            self.timer.start("train/iteration")

            # ---- pass A: rollout without gradients; fixes the Chamfer assignments ----
            eps = torch.randn(T, B, self.num_actions, device=self.device)
            self.env.clear_grad()
            self.env.reset()
            start = self.env.get_checkpoint()
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state(self.device))
            obs_rms = deepcopy(self.obs_rms) if self.obs_rms is not None else None
            feats = []
            with torch.no_grad():
                obs = self._convert_obs(self.env.initialize_trajectory())
                for t in range(T):
                    feats.append(self._feat(obs))
                    if self.obs_rms is not None:
                        for k, v in obs.items():
                            self.obs_rms[k].update(v)
                    obs, rew, done, info = self.env.step(self._policy_actions(obs_rms, obs, eps[t]))
                    obs = self._convert_obs(obs)
                    self.episode_rewards += rew
                    self.episode_lengths += 1
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    if len(ids) > 0:
                        self.episode_rewards_tracker.update(self.episode_rewards[ids])
                        self.episode_lengths_tracker.update(self.episode_lengths[ids])
                        self.num_episodes += len(ids)
                        for i in ids.tolist():
                            self.episode_rewards_hist.append(self.episode_rewards[i].item())
                            self.episode_lengths_hist.append(self.episode_lengths[i].item())
                        self.episode_rewards[ids] = 0.0
                        self.episode_lengths[ids] = 0
                Xraw = torch.stack(feats, 1)  # (B, T, D)
                if os.environ.get("ILD_DEBUG_MEM"):
                    print(f"[ILD] mem after pass A {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
                # running state normaliser, updated with the rollout states (as the paper's state normaliser)
                feat_mean, feat_var, feat_count = self.paper_feat
                m, v, c = Xraw.reshape(-1, Xraw.shape[-1]).mean(0), Xraw.reshape(-1, Xraw.shape[-1]).var(0), B * T
                delta, tot = m - feat_mean, feat_count + c
                feat_mean = feat_mean + delta * c / tot
                feat_var = (feat_var * feat_count + v * c + delta**2 * feat_count * c / tot) / tot
                feat_count = tot
                self.paper_feat = [feat_mean, feat_var, feat_count]
                fstd = feat_var.clamp_min(1e-8).sqrt()
                Y = (Yraw - feat_mean) / fstd
                X = (Xraw - feat_mean) / fstd
                loss_a, best_n, nn_l, nn_t = self._chamfer_assign(X, Y, Ymask, df)
                Yb = Y[best_n]  # (B, L, D) best demo per env
                Lb = Ymask[best_n]  # (B, L)
                n_valid = Lb.sum(1).clamp_min(1)

            # ---- pass B: replay segment by segment with gradients ----
            torch.set_rng_state(rng[0])
            torch.cuda.set_rng_state(rng[1], self.device)
            self.env.clear_grad(start)
            obs = self._convert_obs(self.env.initialize_trajectory())
            optim.zero_grad()
            ar = torch.arange(B, device=self.device)
            drift = 0.0
            for t0 in range(0, T, trunc):
                t1 = min(T, t0 + trunc)
                seg = []
                for t in range(t0, t1):
                    f = (self._feat(obs) - feat_mean) / fstd
                    seg.append(f)
                    if t == t1 - 1:
                        # the next state is where the gradient is cut (as the paper's stop_gradient), so this step
                        # needs no gradient; recording it on DFlex's tape without a backward would leak memory
                        with torch.no_grad():
                            a = self._policy_actions(obs_rms, obs, eps[t])
                            self.env.clear_grad()
                            obs, _, _, _ = self.env.step(a)
                        obs = self._convert_obs(obs)
                        continue
                    a = self._policy_actions(obs_rms, obs, eps[t])
                    if a.requires_grad:
                        a.register_hook(step_norm)
                    obs, _, _, _ = self.env.step(a)
                    obs = self._convert_obs(obs)
                    for v_ in obs.values():
                        if v_.requires_grad:
                            v_.register_hook(step_norm)
                Xs = torch.stack(seg, 1)  # (B, s, D)
                with torch.no_grad():
                    drift = max(drift, (Xs - X[:, t0:t1]).abs().max().item())
                D = Xs.shape[-1]
                # deviation: each rollout state to its assigned demo state
                tgt = Yb[ar[:, None], nn_l[:, t0:t1]]  # (B, s, D)
                dev = ((Xs - tgt) ** 2).mean(-1).clamp_min(1e-12).sqrt().sum() * df / T
                # coverage: each demo state to its assigned rollout state, if that state lies in this segment
                sel = (nn_t >= t0) & (nn_t < t1) & Lb  # (B, L)
                bi, li = sel.nonzero(as_tuple=True)
                cov = Xs.new_zeros(())
                if len(bi) > 0:
                    xs_sel = Xs[bi, nn_t[bi, li] - t0]
                    d = ((xs_sel - Yb[bi, li]) ** 2).mean(-1).clamp_min(1e-12).sqrt()
                    cov = (d / n_valid[bi]).sum()
                loss = (dev + cov) / B
                if loss.requires_grad:
                    loss.backward()
                if os.environ.get("ILD_DEBUG_MEM") and t0 % 100 == 0:
                    print(f"[ILD] mem pass B t={t0} {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
                self.env.clear_grad()  # truncation: cut the graph at the segment boundary
                obs = self._convert_obs(self.env.initialize_trajectory())

            gn = nn.utils.clip_grad_norm_(params, clip)
            if torch.isfinite(gn):
                optim.step()
            else:
                print("non-finite gradient, skipping the update")
            self.agent_steps += per_it
            self.sim_steps += 2 * B * T
            self.timer.end("train/iteration")

            metrics = {
                "train_stats/chamfer_loss": loss_a.item(),
                "train_stats/grad_norm": gn.item(),
                "train_stats/replay_drift": drift,
                "train_stats/lr": lr,
                "train_stats/sim_steps": self.sim_steps,
            }
            mean_rew = -np.inf
            if len(self.episode_rewards_hist) > 0:
                mean_rew = self.episode_rewards_tracker.mean()
                metrics.update({
                    "train_scores/num_episodes": self.num_episodes.item(),
                    "train_scores/episode_rewards": mean_rew,
                    "train_scores/episode_lengths": self.episode_lengths_tracker.mean(),
                })  # fmt: skip
            self.writer.add(self.agent_steps, metrics)
            self.writer.write()
            if len(self.episode_rewards_hist) > 0:
                self._checkpoint_save(mean_rew)
            if self.print_every > 0 and (it + 1) % self.print_every == 0:
                print(
                    f"[ILD] it {it + 1}/{iters} | steps {int(self.agent_steps):,} | chamfer {loss_a.item():.4f} | "
                    f"grad {gn.item():.3f} | drift {drift:.2e} | ep_rewards {mean_rew:.1f}"
                )
        else:
            self.paper_next = iters

        self.protocol_end(self.max_agent_steps)
        self.save(os.path.join(self.ckpt_dir, "final.pth"))
        np.save(open(os.path.join(self.logdir, "ep_rewards_hist.npy"), "wb"), np.array(self.episode_rewards_hist))
        np.save(open(os.path.join(self.logdir, "ep_lengths_hist.npy"), "wb"), np.array(self.episode_lengths_hist))
