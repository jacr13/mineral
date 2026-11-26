import collections
import os
from contextlib import nullcontext
from copy import deepcopy

import torch
import torch.nn as nn

from ...common.demos import get_demos
from ...nets.multi_encoder import MultiEncoder
from ..diffrl.shac import SHAC
from ..diffrl.utils import grad_norm, policy_kl
from .reward import build_focus_loss
from .best_of_k import (
    BestOfK,
    ORCACriterion,
    OTSinkhornCriterion,
    SequenceCosineCriterion,
    SequenceRegressionCriterion,
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
        # "both" relaxes both marginals; "expert" relaxes only the expert side, so every agent step keeps its mass.
        self.loss_ot_marginal_side = self.focus_config.get("loss_ot_marginal_side", "both")
        # Units of the per-step cost (the input of the reward map) and of the window score (the input of the
        # best-of-K softmin). "legacy": as in the reported runs (OT / soft-DTW per-step costs carry a factor 1/T,
        # the L2 ground cost is summed over feature dimensions, the window score sums over steps). "expected":
        # every criterion reports the expected ground cost of each step's match, with the L2 cost averaged over
        # feature dimensions, and the window score is the mean over steps (index-wise matching is the special case
        # of the identity coupling).
        self.loss_cost_units = self.focus_config.get("loss_cost_units", "legacy")
        # Softmin over candidates on scores relative to their mean (scale-free temperature).
        self.loss_best_of_k_relative = bool(self.focus_config.get("loss_best_of_k_relative", False))
        # Reward = phi(c) - c. null: only for the mappings of the reported runs (neg, log_exp, 1_over_1_plus_loss,
        # not exp); true: for every mapping; false: for none.
        self.critic_reward_minus_cost = self.focus_config.get("critic_reward_minus_cost", None)
        # Entropy terms of the SAPO backbone. FOCUS overrides the actor update, so with a SAPO config:
        #   "legacy" (every earlier run): no entropy in the actor's return, the temperature never updated (alpha
        #       stays at init_alpha), while the critic targets go through SHAC's entropy path with the entropy
        #       buffer never filled (zeros) -> after the offset / scale of the SAPO config, a constant +0.5 bonus
        #       per step in the critic targets.
        #   "none": no entropy anywhere; critic targets are FOCUS's own (no constant bonus).
        #   "sapo": SAPO's maximum-entropy objective: alpha * entropy in the actor's discounted return, the same
        #       term in the critic targets, and the temperature learned towards the target entropy.
        self.backbone_entropy = self.focus_config.get("backbone_entropy", "legacy")
        if self.backbone_entropy not in ("legacy", "none", "sapo"):
            raise ValueError(f"Invalid backbone_entropy: {self.backbone_entropy}")
        if self.backbone_entropy == "sapo" and not (self.with_logprobs and self.with_autoent):
            raise ValueError('backbone_entropy "sapo" needs a SAPO config (with_logprobs, with_autoent)')
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
        # Feature dimensions the imitation cost is computed on (null = all). E.g. positions only, without
        # velocities: then the order of the states is the only cue for the direction of motion (ordering test).
        _dims = self.focus_config.get("loss_feature_dims", None)
        self.loss_feature_dims = None if _dims in (None, "null", "none") else [int(d) for d in _dims]
        self.actor_grad_estimator = self.focus_config.get("actor_grad_estimator", "pathwise")
        # Bootstrap of the actor's return (writer request R19, 2026-11-25). "next_state": the value of s_{i+1} after the
        # action of step i, zero after a fall and the critic's value of the terminal state at a time-out, for the min
        # and the average of the critics alike (as SHAC.compute_actor_loss). "as_run": the objective of the runs up to
        # commit 22e2055, which indexed the (H+1)-row value tensor with the H-row index, so it bootstrapped from the
        # value of s_i (the state before the last action), and never zeroed the average-critic value after a fall.
        self.actor_bootstrap = self.focus_config.get("actor_bootstrap", "next_state")
        if self.actor_bootstrap not in ("next_state", "as_run"):
            raise ValueError(f"Invalid actor_bootstrap: {self.actor_bootstrap}")
        if self.actor_grad_estimator not in ("pathwise", "score_function"):
            raise ValueError(f"Invalid actor_grad_estimator: {self.actor_grad_estimator}")
        # Ordered-alignment (soft-DTW) alternative to the Sinkhorn coupling.
        self.loss_dtw_gamma = float(self.focus_config.get("loss_dtw_gamma", 0.1))
        self.loss_dtw_normalize = self.focus_config.get("loss_dtw_normalize", "rows")
        # ORCA-style ordered coverage (reviewer Q9cS's named baseline).
        self.loss_orca_lam = float(self.focus_config.get("loss_orca_lam", 0.25))
        self.loss_orca_cost_mode = self.focus_config.get("loss_orca_cost_mode", "neg_log_r")
        self.loss_orca_normalize_p = bool(self.focus_config.get("loss_orca_normalize_p", False))
        self.loss_orca_normalize_cost = bool(self.focus_config.get("loss_orca_normalize_cost", True))
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

        if self.imitation_loss_type == "index":
            # no alignment: step t of the agent window is matched to step t of the candidate segment, with the
            # configured ground cost (the same cost as the aligned variants)
            self.imitation_loss_type = "cosine" if self.loss_ot_cost_type == "cosine" else "l2"
        if self.imitation_loss_type in ["ot", "l2", "cosine", "softdtw", "softdtw_subseq", "orca"]:
            obs_dim = self.num_obs
            if self.network_config.get("encoder", None) is not None:
                obs_dim = self.encoder.out_dim
            if self.loss_feature_dims is not None:
                obs_dim = len(self.loss_feature_dims)
            # one builder for FOCUS and its zeroth-order variants (reward.build_focus_loss)
            cfg = dict(self.focus_config)
            cfg["imitation_loss_type"] = self.imitation_loss_type
            self.loss_fn = build_focus_loss(cfg, self.horizon_len, obs_dim, self.device)
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
                info["diag_actor_grad_nonfinite"] = sum(
                    (~torch.isfinite(q.grad)).sum() for q in self.actor.parameters() if q.grad is not None).float()
                info["diag_actor_loss_nonfinite"] = (~torch.isfinite(actor_loss.detach())).float()
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
                "per_step_cost_mean",
                "per_step_cost_std",
                "reward_dead_frac",
                "k_eff",
                "row_mass_min",
                "row_mass_max",
                "row_mass_cv",
                "entropy",
                "diag_envs_nonfinite_obs",
                "diag_envs_exploding_obs",
                "diag_envs_nonfinite_cost",
                "diag_envs_nonfinite_cost_finite_obs",
                "diag_steps_nonfinite_reward",
                "diag_demo_feat_nonfinite",
                "diag_actor_grad_nonfinite",
                "diag_actor_loss_nonfinite",
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

        if self.backbone_entropy == "sapo":
            # temperature update, as SHAC.update_actor
            alpha = self.get_alpha(detach=False)
            if self.unscale_entropy_alpha and self.scale_by_target_entropy:
                alpha = alpha * abs(self.target_entropy)
            alpha_loss = (alpha * (self._entropy - self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            if self.shac_config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.alpha_optim.param_groups[0]["params"], self.shac_config.max_grad_norm)
            self.alpha_optim.step()
            results["entropy_alpha_loss"].append(alpha_loss.detach())

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

    def _encode_window(self, obs):
        """Encode a window of observations {key: (B, T, ...)} into features (B, T, Z).

        The default encoder (x['obs']) acts element-wise and is applied as is. A MultiEncoder (Rewarped tasks: point
        clouds, flattened MLP inputs) expects one frame per row, so the window is flattened to (B * T, ...), encoded,
        and its feature 'z' reshaped back.
        """
        if not isinstance(self.actor_encoder, MultiEncoder):
            return self.actor_encoder(obs)
        B, T = next(iter(obs.values())).shape[:2]
        out = self.actor_encoder({k: v.reshape(B * T, *v.shape[2:]) for k, v in obs.items()})
        z = out["z"] if isinstance(out, dict) else out
        return z.view(B, T, *z.shape[1:])

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
        sapo = self.backbone_entropy == "sapo"
        step_entropy = []

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
            if sapo:
                logprob = distr.log_prob(actions).sum(dim=-1)
                distr_ent = distr.entropy().sum(dim=-1)
                step_entropy.append(distr_ent if self.use_distr_ent else -1.0 * logprob)
                with torch.no_grad():
                    self.logprobs[i] = logprob.detach()
                    self.distr_ent[i] = distr_ent.detach()

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
                        if self.actor_bootstrap == "next_state":
                            avg_next_values[i + 1, done_env_id] = 0.0
                    else:
                        real_z_target = self.encoder_target(real_obs_term)
                        if self.actor_bootstrap == "next_state":
                            real_next_values, avg_real_next_values = self.critic_target(
                                real_z_target, return_type="min_and_avg"
                            )
                            next_values[i + 1, done_env_id] = real_next_values.squeeze(-1)
                            avg_next_values[i + 1, done_env_id] = (
                                avg_real_next_values.squeeze(-1) if self.actor_loss_avgcritics
                                else real_next_values.squeeze(-1)
                            )
                        else:
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

        obs_z = self._encode_window(obs_window)
        exp_z = self._encode_window(obs_exp).detach()
        if self.loss_feature_dims is not None:  # match on these feature dimensions only (actor and critic see all)
            obs_z = obs_z[..., self.loss_feature_dims]
            exp_z = exp_z[..., self.loss_feature_dims]

        self.timer.start("train/compute_actor_loss/loss_fn")
        loss, info = self.loss_fn(obs_z, exp_z, sim_is_window=True)
        self.timer.end("train/compute_actor_loss/loss_fn")

        if "per_step_costs" not in info:
            raise ValueError("BestOfK info missing per_step_costs for FOCUS")
        self._maybe_save_plan_heatmaps(info)
        per_step_costs = info.pop("per_step_costs")  # [B,T]
        per_step_rewards = self._build_pseudo_rewards(per_step_costs)
        with torch.no_grad():  # diagnostics of non-finite epochs (logging only, nothing below changes training)
            obs_ok = torch.stack([torch.isfinite(v).reshape(v.shape[0], v.shape[1], -1).all(-1)
                                  for v in obs_window.values()]).all(0)  # [B,T]
            obs_big = torch.stack([(v.abs() > 1e6).reshape(v.shape[0], v.shape[1], -1).any(-1)
                                   for v in obs_window.values()]).any(0)
            cost_bad = ~torch.isfinite(per_step_costs)
            info["diag_envs_nonfinite_obs"] = (~obs_ok).any(1).float().sum()
            info["diag_envs_exploding_obs"] = (obs_big & obs_ok).any(1).float().sum()
            info["diag_envs_nonfinite_cost"] = cost_bad.any(1).float().sum()
            # non-finite cost although every state of the env's window is finite: the OT computation, not the simulator
            info["diag_envs_nonfinite_cost_finite_obs"] = (cost_bad.any(1) & obs_ok.all(1)).float().sum()
            info["diag_steps_nonfinite_reward"] = (~torch.isfinite(per_step_rewards)).float().sum()
            info["diag_demo_feat_nonfinite"] = (~torch.isfinite(exp_z)).float().sum()
        info["pseudo_reward_mean"] = per_step_rewards.detach().mean()
        info["pseudo_reward_std"] = per_step_rewards.detach().std(unbiased=False)
        info["imitation_loss"] = loss.detach()
        info["per_step_cost_mean"] = per_step_costs.detach().mean()
        info["per_step_cost_std"] = per_step_costs.detach().std(unbiased=False)
        # share of steps where exp(-c) has underflowed to nothing (no learning signal from an exp-type map)
        info["reward_dead_frac"] = (per_step_costs.detach() > 13.8).float().mean()

        step_rewards = per_step_rewards.transpose(0, 1)  # [T,B]
        # step_cost = per_step_costs.transpose(0, 1)  # [T,B]

        with torch.no_grad():
            self.rew_buf.copy_(step_rewards.detach())
        # use live next_values for actor gradients; buffers stay detached for critic use
        next_vs_live = avg_next_values if self.actor_loss_avgcritics else next_values
        if self.actor_bootstrap == "next_state":
            # row i = value after the action of step i, the indexing of the H-row buffers (done_mask, next_values)
            next_vs_live = next_vs_live[1:]

        actor_rewards = step_rewards
        if sapo:
            # SAPO: alpha * (offset / scaled) entropy in the discounted return, as SHAC.compute_actor_loss does
            ent = torch.stack(step_entropy)  # [T, B], differentiable
            ent_t = ent
            if self.offset_by_target_entropy:
                ent_t = (ent_t + abs(self.target_entropy)) * 0.5
            if self.scale_by_target_entropy:
                ent_t = ent_t * (1.0 / abs(self.target_entropy))
            actor_rewards = step_rewards + self.get_alpha(scalar=True) * ent_t
            self._entropy = ent.detach().mean(dim=0)  # per env, as SHAC (sum over the horizon / horizon)
            info["entropy"] = ent.detach().mean()

        returns = self._compute_returns_from_rewards(
            step_rewards=actor_rewards,
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
        if self.critic_reward_minus_cost is None:
            minus_cost = self.critic_reward_shapping and self.critic_reward_mapping in [
                "neg",
                "log_exp",
                "1_over_1_plus_loss",
            ]
        else:
            minus_cost = bool(self.critic_reward_minus_cost)
        if minus_cost:
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

    def compute_target_values_with_entropy(self):
        """SAPO configs route critic targets here (entropy_in_targets). See backbone_entropy."""
        if self.backbone_entropy == "none":
            return self.compute_target_values()
        return super().compute_target_values_with_entropy()

    def compute_target_values(self):
        """TD(lambda) critic targets: SHAC's recursion, which stops at episode ends through done_mask (the next value
        is zero after a termination and the critic's value after a time-out, as set in the rollout).

        The earlier override here computed r + gamma * (lambda * G' + V(s')) -- the full V(s') at every step instead of
        (1 - lambda) * V(s') -- and dropped the bootstrap at the end of the window: on a constant-reward check it gives
        156 where the lambda-return is 23 (tests/test_focus_criteria.py). It was used by every FOCUS run whose config
        does not route targets through the entropy path, i.e. every SHAC-base run (SAPO-base runs never called it).
        """
        return SHAC.compute_target_values(self)
