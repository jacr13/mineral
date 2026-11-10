import collections
import os
from contextlib import nullcontext
from copy import deepcopy

import torch
import torch.nn as nn

from ...common.demos import get_demos
from ..diffrl.shac import SHAC
from ..diffrl.utils import grad_norm, policy_kl
from .best_of_k import (
    BestOfK,
    OTSinkhornCriterion,
    SequenceCosineCriterion,
    SequenceRegressionCriterion,
    ORCACriterion,
    SoftDTWCriterion,
)


class FOCUS(SHAC):
    """FOCUS variant that reuses SHAC base functionality."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        self.focus_config = full_cfg.agent.get("focus", {})
        self.actor_value_return_type = self.focus_config.get("actor_value_return_type", "min")

        self.imitation_loss_type = self.focus_config.get("imitation_loss_type", "ot")
        self.input_type = self.focus_config.get("input_type", "state")

        self.critic_reward_shapping = self.focus_config.get("critic_reward_shapping", True)
        self.critic_disabled = self.focus_config.get("critic_disabled", False)
        self.critic_reward_mapping = self.focus_config.get("critic_reward_mapping", "log_exp")
        self.critic_reward_scale = self.focus_config.get("critic_reward_scale", 1.0)
        self.critic_reward_normalize = self.focus_config.get("critic_reward_normalize", False)

        demos_config = self.focus_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)

        # Sinkhorn settings and the best-of-K softmin temperature are read from
        # the config here; the defaults are the values every reported run used
        # (they were previously hardcoded below, so loss_ot_eps/loss_ot_iters in
        # the configs were silently ignored on this path).
        self.loss_ot_eps = float(self.focus_config.get("loss_ot_eps", 0.1))
        self.loss_ot_iters = int(self.focus_config.get("loss_ot_iters", 60))
        self.loss_ot_huber_delta = float(self.focus_config.get("loss_ot_huber_delta", 1.0))
        self.loss_best_of_k_tau = float(self.focus_config.get("loss_best_of_k_tau", 0.5))
        # False differentiates through the Sinkhorn iterations instead of using
        # the fixed-coupling surrogate; True is what every reported run used.
        self.loss_ot_detach_plan = bool(self.focus_config.get("loss_ot_detach_plan", True))
        # null = balanced OT with uniform marginals (as published). A float
        # relaxes the marginals (unbalanced OT), which is the fix reviewer Q9cS
        # suggested for the fragmented plans in Figs. 3(c)/9(f)(h).
        _rho = self.focus_config.get("loss_ot_marginal_rho", None)
        self.loss_ot_marginal_rho = None if _rho in (None, "null", "none") else float(_rho)
        # "pathwise"       : differentiate the return through the simulator -- FOCUS as published.
        # "score_function" : identical objective, reward, critic, architecture and
        #                    horizon, but the gradient is estimated with REINFORCE
        #                    and the critic as baseline instead of the dynamics
        #                    Jacobian. This isolates the simulator-derivative
        #                    contribution, which the ZO-FOCUS comparison cannot do
        #                    because it swaps the whole actor-critic for PPO/SAC.
        #                    Simply detaching observations is NOT an option: the
        #                    objective depends on the policy only through the
        #                    dynamics, so the actor would receive no gradient.
        self.actor_grad_estimator = self.focus_config.get("actor_grad_estimator", "pathwise")
        if self.actor_grad_estimator not in ("pathwise", "score_function"):
            raise ValueError(f"Invalid actor_grad_estimator: {self.actor_grad_estimator}")
        # Ordered-alignment (soft-DTW) alternative to the Sinkhorn coupling.
        self.loss_dtw_gamma = float(self.focus_config.get("loss_dtw_gamma", 0.1))
        self.loss_dtw_normalize = self.focus_config.get("loss_dtw_normalize", "rows")
        # ORCA-style ordered coverage (reviewer Q9cS's named baseline).
        self.loss_orca_lam = float(self.focus_config.get("loss_orca_lam", 0.25))
        self.loss_orca_cost_mode = self.focus_config.get("loss_orca_cost_mode", "neg_log_r")
        self.loss_orca_normalize_p = bool(self.focus_config.get("loss_orca_normalize_p", False))
        self.loss_ot_cost_type = self.focus_config.get("loss_ot_cost_type", "l2")
        self.loss_use_huber_speedup = self.focus_config.get("loss_use_huber_speedup", True)
        self.loss_mlp_features_dim = self.focus_config.get("loss_mlp_features_dim", None)
        self.loss_use_detached_prev_obs = self.focus_config.get("loss_use_detached_prev_obs", True)
        self.loss_best_of_k_k = self.focus_config.get("loss_best_of_k_k", 8)
        self.save_plan_heatmaps = self.focus_config.get("save_plan_heatmaps", False)
        self.save_plan_heatmaps_every = int(self.focus_config.get("save_plan_heatmaps_every", 50_000))
        self.save_plan_heatmaps_num_samples = int(self.focus_config.get("save_plan_heatmaps_num_samples", 5))
        self._last_plan_heatmap_step = None
        self._plan_heatmap_plt = None
        self.plan_heatmap_dir = os.path.join(self.logdir, "ot_plan_heatmaps")
        if self.save_plan_heatmaps and (not self.multi_gpu or self.rank == 0):
            os.makedirs(self.plan_heatmap_dir, exist_ok=True)

        if self.imitation_loss_type in ["ot", "l2", "cosine", "softdtw", "softdtw_subseq", "orca"]:
            if self.imitation_loss_type == "ot":
                ot_params = dict(
                    eps=self.loss_ot_eps,
                    iters=self.loss_ot_iters,
                    cost_type=self.loss_ot_cost_type,
                    huber_delta=self.loss_ot_huber_delta,
                    use_huber_speedup=self.loss_use_huber_speedup,
                    detach_plan=self.loss_ot_detach_plan,
                    marginal_rho=self.loss_ot_marginal_rho,
                )
                criterion = OTSinkhornCriterion(**ot_params)
            elif self.imitation_loss_type in ("softdtw", "softdtw_subseq"):
                # Same ground cost and the same per-step cost scale as the OT
                # criterion (see SoftDTWCriterion); only the alignment differs.
                subseq = self.imitation_loss_type == "softdtw_subseq"
                criterion = SoftDTWCriterion(
                    gamma=self.loss_dtw_gamma,
                    cost_type=self.loss_ot_cost_type,
                    huber_delta=self.loss_ot_huber_delta,
                    subsequence=subseq,
                    open_end=subseq,
                    normalize=self.loss_dtw_normalize,
                )
            elif self.imitation_loss_type == "orca":
                criterion = ORCACriterion(
                    lam=self.loss_orca_lam,
                    cost_type=self.loss_ot_cost_type,
                    huber_delta=self.loss_ot_huber_delta,
                    normalize_p=self.loss_orca_normalize_p,
                    cost_mode=self.loss_orca_cost_mode,
                )
            elif self.imitation_loss_type == "l2":
                criterion = SequenceRegressionCriterion(use_huber=False, reduction="mean")
            elif self.imitation_loss_type == "cosine":
                criterion = SequenceCosineCriterion()
            else:
                raise NotImplementedError(self.imitation_loss_type)

            obs_dim = self.num_obs
            if self.network_config.get("encoder", None) is not None:
                obs_dim = self.encoder.out_dim

            bok_params = dict(
                T=self.horizon_len if self.input_type == "state" else self.horizon_len + 1,
                K=self.loss_best_of_k_k,
                tau=self.loss_best_of_k_tau,
                input_type=self.input_type,
                detach_prev_obs=self.loss_use_detached_prev_obs,
                mlp_features_dim=self.loss_mlp_features_dim,
                feature_dim=obs_dim,
                return_per_step_costs=True,
                device=self.device,
            )
            self.loss_fn = BestOfK(**bok_params, criterion=criterion)
        elif self.imitation_loss_type == "chamfer":
            from .chamfer_loss import ChamferImitationLoss

            self.loss_fn = ChamferImitationLoss()
        else:
            raise NotImplementedError("Adversarial imitation not yet supported for FOCUS")

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
                "Lk_min",
                "Lk_mean",
                "pseudo_reward_mean",
                "pseudo_reward_std",
                "imitation_loss",
                "returns_mean",
                "expert_return",
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
        # The score-function estimator never differentiates through the
        # simulator, so building the pathwise graph over the rollout is not just
        # wasted work -- nothing ever backprops through it, so it is not freed
        # by actor_loss.backward() and memory climbs every epoch until CUDA OOM
        # (observed at epoch ~82 on Hopper, ~10 on SNU Humanoid, whose actor is
        # larger). Roll out under no_grad in that mode; the gradient comes
        # solely from recomputing log pi(a|s) in _score_function_actor_loss.
        rollout_ctx = torch.no_grad() if self.actor_grad_estimator == "score_function" else nullcontext()
        with rollout_ctx:
            return self._compute_actor_loss_inner()

    def _compute_actor_loss_inner(self):
        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = deepcopy(self.obs_rms)

        obs = self.env.initialize_trajectory()
        obs = self._convert_obs(obs)

        obs_window = {k: [] for k in obs.keys()}
        if self.input_type == "state_state":
            obs_window = {k: [v] for k, v in obs.items()}

        next_values = torch.zeros((self.horizon_len + 1, self.num_envs), dtype=torch.float32, device=self.device)
        avg_next_values = torch.zeros_like(next_values)

        if self.obs_rms is not None:
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

            self.timer.start("train/compute_actor_loss/forward_sim")
            obs, rew, done, info = self.env.step(actions)
            self.timer.end("train/compute_actor_loss/forward_sim")

            real_obs = info.get("obs_before_reset", obs)
            real_obs = obs if real_obs is None else real_obs

            obs = self._convert_obs(obs)
            real_obs = self._convert_obs(real_obs)

            with torch.no_grad():
                raw_rew = rew.clone()

            with torch.no_grad():
                self.episode_rewards += raw_rew
                self.episode_discounted_rewards += self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                self.episode_lengths += 1

            if self.obs_rms is not None:
                with torch.no_grad():
                    for k, v in obs.items():
                        self.obs_rms[k].update(v)
                obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}
                real_obs = {k: obs_rms[k].normalize(v) for k, v in real_obs.items()}

            z_target = self.encoder_target(obs)
            if self.actor_loss_avgcritics:
                pred_val, avg_pred_val = self.critic_target(z_target, return_type="min_and_avg")
                pred_val = pred_val.squeeze(-1)
                avg_pred_val = avg_pred_val.squeeze(-1)
            else:
                pred_val = self.critic_target(z_target, return_type="min").squeeze(-1)
                avg_pred_val = pred_val
            next_values[i + 1] = pred_val
            avg_next_values[i + 1] = avg_pred_val

            for k, v in real_obs.items():
                obs_window[k].append(v)

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                done_env_ids = done_env_ids.detach().cpu()
                terminal_obs = info.get("obs_before_reset", None)
                if terminal_obs is None:
                    terminal_obs = obs
                terminal_obs = self._convert_obs(terminal_obs)
                if self.obs_rms is not None:
                    terminal_obs = {k: obs_rms[k].normalize(v) for k, v in terminal_obs.items()}
                self.episode_rewards_tracker.update(self.episode_rewards[done_env_ids])
                self.episode_discounted_rewards_tracker.update(self.episode_discounted_rewards[done_env_ids])
                self.episode_lengths_tracker.update(self.episode_lengths[done_env_ids])
                self.num_episodes += len(done_env_ids)
                for done_env_id in done_env_ids:
                    self.episode_rewards_hist.append(self.episode_rewards[done_env_id].item())
                    self.episode_discounted_rewards_hist.append(self.episode_discounted_rewards[done_env_id].item())
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
                    self.episode_discounted_rewards[done_env_id] = 0.0
                    self.episode_lengths[done_env_id] = 0
                    self.episode_gamma[done_env_id] = 1.0

            if i < self.horizon_len - 1:
                self.done_mask[i] = done.to(dtype=torch.float32)
            else:
                self.done_mask[i] = torch.ones_like(done, dtype=torch.float32)
            # detach for buffers to avoid graph reuse in later critic updates
            self.next_values[i] = next_values[i + 1].detach()
            self.avg_next_values[i] = avg_next_values[i + 1].detach()

        obs_window = {k: torch.stack(v, dim=1) for k, v in obs_window.items()}
        with torch.no_grad():
            if self.obs_rms is not None:
                obs_exp = {k: obs_rms[k].normalize(v) for k, v in self.demos["obs"].items()}
            else:
                obs_exp = self.demos["obs"]

        # B_agent, T_agent = next(iter(obs_window.values())).shape[0:2]
        # B_exp, T_exp = next(iter(obs_exp.values())).shape[0:2]

        # obs_window = {k: v.view(-1, *v.shape[2:]) for k, v in obs_window.items()}
        # obs_exp = {k: v.view(-1, *v.shape[2:]) for k, v in obs_exp.items()}

        obs_z = self.actor_encoder(obs_window)
        exp_z = self.actor_encoder(obs_exp).detach()

        # if isinstance(obs_z, dict):
        #     obs_z = obs_z["z"].view(B_agent, T_agent, *obs_z["z"].shape[1:])
        # if isinstance(exp_z, dict):
        #     exp_z = exp_z["z"].view(B_exp, T_exp, *exp_z["z"].shape[1:])

        # exp_z = exp_z.detach()

        self.timer.start("train/compute_actor_loss/loss_fn")
        loss, info = self.loss_fn(obs_z, exp_z, sim_is_window=True)
        self.timer.end("train/compute_actor_loss/loss_fn")

        if "per_step_costs" not in info:
            raise ValueError("BestOfK info missing per_step_costs for FOCUS")
        self._maybe_save_plan_heatmaps(info)
        per_step_costs = info.pop("per_step_costs")  # [B,T]
        per_step_rewards = self._build_pseudo_rewards(per_step_costs)
        info["pseudo_reward_mean"] = per_step_rewards.detach().mean()
        info["pseudo_reward_std"] = per_step_rewards.detach().std(unbiased=False)
        info["imitation_loss"] = loss.detach()
        info["per_step_cost_mean"] = per_step_costs.detach().mean()
        info["per_step_cost_std"] = per_step_costs.detach().std(unbiased=False)

        step_rewards = per_step_rewards.transpose(0, 1)  # [T,B]
        # step_cost = per_step_costs.transpose(0, 1)  # [T,B]

        with torch.no_grad():
            self.rew_buf.copy_(step_rewards.detach())
        # use live next_values for actor gradients; buffers stay detached for critic use
        next_vs_live = avg_next_values if self.actor_loss_avgcritics else next_values

        returns = self._compute_returns_from_rewards(
            step_rewards=step_rewards,
            next_vs=next_vs_live,
        )
        info["returns_mean"] = returns.detach().mean()

        info["expert_return"] = self.demos["expert_return"]

        if self.actor_grad_estimator == "pathwise":
            actor_loss = -returns.mean()
        else:
            # Re-enable grad: the enclosing no_grad covers the rollout only.
            # obs_buf/action_buf are detached stores, so this builds a graph
            # spanning just the actor, which is what we want to differentiate.
            with torch.enable_grad():
                actor_loss = self._score_function_actor_loss(step_rewards, next_vs_live, info)

        self.agent_steps += self.horizon_len * self.num_envs
        return actor_loss, info

    def _score_function_actor_loss(self, step_rewards, next_vs, info):
        """REINFORCE estimate of the same objective, with the critic as baseline.

        The pathwise estimator differentiates the return through the simulator.
        Here the return is treated as a constant and the gradient comes from
        the log-probabilities of the actions that were actually taken:

            grad J  ~  E[ sum_t grad log pi(a_t|s_t) * (G_t - V(s_t)) ]

        with G_t the discounted return-to-go built from the *same* per-step OT
        rewards and bootstrapped by the *same* critic. Everything outside the
        estimator -- reward, critic, network, horizon, best-of-K -- is shared
        with the pathwise path, so a difference in outcome is attributable to
        the estimator.

        The critic baseline matters: plain REINFORCE would mostly demonstrate
        that an unbaselined score-function estimator is high-variance, which is
        not the question being asked.
        """
        rewards = step_rewards.detach()
        next_vs = next_vs.detach()

        # discounted return-to-go, stopping at episode boundaries exactly as
        # _compute_returns_from_rewards does
        H = self.horizon_len
        G = torch.zeros_like(rewards)
        running = next_vs[H - 1].clone()
        for i in reversed(range(H)):
            done = self.done_mask[i]
            boot = (1.0 - done) * running + done * next_vs[i]
            G[i] = rewards[i] + self.gamma * boot
            running = G[i]

        with torch.no_grad():
            obs = {k: v.view(-1, *v.shape[2:]) for k, v in self.obs_buf.items()}
            z = self.encoder_target(obs)
            values = self.critic_target(z, return_type="min").squeeze(-1).view(H, self.num_envs)
        advantage = G - values
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)

        # Recompute log pi(a_t|s_t) for the stored actions, with grad to theta.
        #
        # The stored actions come from a SquashedNormal, so log_prob inverts the
        # tanh. mineral's TanhTransform._inverse calls atanh without clamping
        # (it relies on cache_size=1, which only helps when log_prob is called
        # on the very tensor rsample returned). Here the actions are clones fed
        # to a freshly built distribution, so the cache misses and any action
        # saturated to +-1 in float32 gives atanh = +-inf -> NaN gradients.
        # Clamping into the open interval is the standard remedy.
        eps = 1e-6
        logp = []
        for i in range(H):
            obs_i = {k: v[i] for k, v in self.obs_buf.items()}
            _, _, distr = self.actor(self.actor_encoder(obs_i))
            a = self.action_buf[i].clamp(-1.0 + eps, 1.0 - eps)
            logp.append(distr.log_prob(a).sum(-1))
        logp = torch.stack(logp)
        if not torch.isfinite(logp).all():
            n_bad = int((~torch.isfinite(logp)).sum())
            logp = torch.nan_to_num(logp, nan=0.0, posinf=0.0, neginf=0.0)
            info["score_function_nonfinite_logp"] = torch.tensor(float(n_bad))

        info["score_function_adv_std"] = advantage.std(unbiased=False).detach()
        return -(logp * advantage).sum(dim=0).mean()

    def _maybe_save_plan_heatmaps(self, info):
        if not self.save_plan_heatmaps:
            return
        if self.multi_gpu and self.rank != 0:
            return
        if self._last_plan_heatmap_step is not None:
            if self.agent_steps - self._last_plan_heatmap_step < self.save_plan_heatmaps_every:
                return

        transport_plans = info.get("transport_plans")
        if transport_plans is None:
            return

        plt = self._get_plan_heatmap_plt()
        if plt is None:
            return

        plans = transport_plans.detach().to(device="cpu", dtype=torch.float32)
        starts = info["starts"].detach().to(device="cpu", dtype=torch.long)
        losses = info["Lk"].detach().to(device="cpu", dtype=torch.float32)
        best_k = info["best_idx"]
        expert_ids = info.get("expert_ids")
        if expert_ids is not None:
            expert_ids = expert_ids.detach().to(device="cpu", dtype=torch.long)

        num_samples = min(self.save_plan_heatmaps_num_samples, plans.shape[0])
        losses_dir = os.path.join(self.logdir, "losses")
        os.makedirs(losses_dir, exist_ok=True)

        for sample_idx in range(num_samples):
            with open(os.path.join(losses_dir, f"{sample_idx}.csv"), "a+") as f:
                f.write(f"{','.join(map(str, losses[sample_idx].tolist()))}\n")

            for k_idx in range(plans.shape[1]):
                plan = plans[sample_idx, k_idx].numpy()
                fig, ax = plt.subplots(figsize=(5, 4.5))
                im = ax.imshow(
                    plan,
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    cmap=self._plan_heatmap_mpl.colormaps["Blues"],
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xlabel("Expert timestep")
                ax.set_ylabel("Sim timestep")

                title_parts = [
                    f"step={int(self.agent_steps)}",
                    f"k={k_idx}",
                    f"loss={losses[sample_idx, k_idx].item():.4f}",
                ]
                # if expert_ids is not None:
                #     title_parts.append(f"expert={int(expert_ids[sample_idx, k_idx])}")
                ax.set_title(" | ".join(title_parts))

                filename_parts = [
                    f"step_{int(self.agent_steps):012d}",
                    f"k_{k_idx:02d}",
                    f"start_{int(starts[sample_idx, k_idx]):04d}",
                ]
                if expert_ids is not None:
                    filename_parts.append(f"expert_{int(expert_ids[sample_idx, k_idx]):04d}")

                if k_idx == best_k[sample_idx]:
                    filename_parts.append("best_k")

                out_path = os.path.join(self.plan_heatmap_dir, f"sample_{sample_idx:02d}", "__".join(filename_parts) + ".png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                fig.tight_layout()
                fig.savefig(out_path, dpi=150)
                plt.close(fig)

        self._last_plan_heatmap_step = self.agent_steps

    def _get_plan_heatmap_plt(self):
        if self._plan_heatmap_plt is not None:
            return self._plan_heatmap_plt

        try:
            import matplotlib as mpl

            mpl.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"Skipping OT plan heatmaps because matplotlib is unavailable: {exc}")
            self._plan_heatmap_mpl = None
            self._plan_heatmap_plt = None
            return None

        self._plan_heatmap_mpl = mpl
        self._plan_heatmap_plt = plt
        return self._plan_heatmap_plt

    def _build_pseudo_rewards(self, per_step_costs: torch.Tensor) -> torch.Tensor:
        if self.critic_reward_mapping == "neg":
            rewards = -per_step_costs
        elif self.critic_reward_mapping == "exp":
            rewards = torch.exp(-per_step_costs)
        elif self.critic_reward_mapping == "log":
            rewards = -torch.log(per_step_costs + 1e-7)
        elif self.critic_reward_mapping == "log_exp":
            rewards = -torch.log(1 - torch.exp(-per_step_costs) + 1e-7)
        elif self.critic_reward_mapping == "1_over_1_plus_loss":
            rewards = 1.0 / (1.0 + per_step_costs)
        else:
            raise NotImplementedError(self.critic_reward_mapping)

        # allows gradients to flow through cost when loss is high and exp goes to 0.
        # NOTE: this makes the reward phi(L) - L, not phi(L), for these mappings.
        if self.critic_reward_shapping and self.critic_reward_mapping in [
            "neg",
            "log_exp",
            "1_over_1_plus_loss",
        ]:
            rewards = rewards - per_step_costs

        if self.critic_reward_normalize:
            mean = rewards.mean(dim=1, keepdim=True)
            std = rewards.std(dim=1, keepdim=True, unbiased=False)
            rewards = (rewards - mean) / (std + 1e-6)
        rewards = self.reward_shaper(rewards) * self.critic_reward_scale
        return rewards

    def _compute_returns_from_rewards(
        self,
        step_rewards: torch.Tensor,
        next_vs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute returns from per-step costs and rewards, stopping bootstrapping at window boundaries."""
        rewards = step_rewards

        if self.critic_disabled:
            return rewards.sum(dim=0)

        # rewards: [T, B]
        returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        rew_acc = torch.zeros_like(returns)
        gamma = torch.ones_like(returns)
        if next_vs is None:
            next_vs = self.avg_next_values if self.actor_loss_avgcritics else self.next_values

        for i in range(self.horizon_len):
            rew_acc = rew_acc + gamma * rewards[i]
            done = self.done_mask[i]
            if i < self.horizon_len - 1:
                term = rew_acc * done + self.gamma * gamma * next_vs[i] * done
                returns = returns + term
            else:
                term = rew_acc + self.gamma * gamma * next_vs[i]
                returns = returns + term
            gamma = gamma * self.gamma
            not_done = 1.0 - done
            rew_acc = rew_acc * not_done
            gamma = gamma * not_done + torch.ones_like(gamma) * done
        return returns

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
        """Override SHAC target computation to stop bootstrapping across episode boundaries."""
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
