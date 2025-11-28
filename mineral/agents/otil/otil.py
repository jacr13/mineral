import collections
from copy import deepcopy
import torch
import torch.nn as nn

from ...common.demos import get_demos
from ..diffrl.shac import SHAC
from ..diffrl.utils import grad_norm, policy_kl
from .best_of_k import BestOfK, BestOfKConfig, OTSinkhornCriterion, SequenceCosineCriterion, SequenceRegressionCriterion


class OTIL(SHAC):
    """OTIL variant that reuses SHAC base functionality."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        self.otil_config = full_cfg.agent.get("otil", {})
        self.actor_loss_coef = self.otil_config.get("actor_loss_coef", 1.0)
        self.actor_value_coef = self.otil_config.get("actor_value_coef", 0.0)
        self.actor_value_return_type = self.otil_config.get("actor_value_return_type", "min")

        self.imitation_loss_type = self.otil_config.get("imitation_loss_type", "ot")

        self.critic_reward_reduction = self.otil_config.get("critic_reward_reduction", "min")
        self.critic_reward_scale = self.otil_config.get("critic_reward_scale", 1.0)
        self.critic_reward_tau = self.otil_config.get("critic_reward_tau", 0.5)
        self.critic_reward_normalize = self.otil_config.get("critic_reward_normalize", False)

        demos_config = self.otil_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)

        cfg_bok = BestOfKConfig(
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
            return_per_step_costs=True,
        )

        if self.imitation_loss_type == "ot":
            criterion = OTSinkhornCriterion(
                eps=cfg_bok.eps,
                iters=cfg_bok.sinkhorn_iters,
                use_huber=cfg_bok.use_huber,
                huber_delta=cfg_bok.huber_delta,
            )
        elif self.imitation_loss_type == "l2":
            criterion = SequenceRegressionCriterion(use_huber=False, reduction="mean")
        elif self.imitation_loss_type == "cosine":
            criterion = SequenceCosineCriterion()
        else:
            raise NotImplementedError(self.imitation_loss_type)

        self.loss_fn = BestOfK(cfg_bok, criterion=criterion, device=self.device)

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
            if self.obs_rms is not None:
                obs_rms = deepcopy(self.obs_rms)

        obs = self.env.initialize_trajectory()
        obs = self._convert_obs(obs)

        obs_window = {k: [] for k in obs.keys()}
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

            obs, rew, done, info = self.env.step(actions)
            real_obs = info.get("obs_before_reset", obs)
            real_obs = obs if real_obs is None else real_obs

            obs = self._convert_obs(obs)
            real_obs = self._convert_obs(real_obs)

            with torch.no_grad():
                raw_rew = rew.clone()

            with torch.no_grad():
                self.episode_rewards += raw_rew
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
                self.episode_lengths_tracker.update(self.episode_lengths[done_env_ids])
                self.num_episodes += len(done_env_ids)
                for done_env_id in done_env_ids:
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

        obs_z = self.actor_encoder(obs_window)
        exp_z = self.actor_encoder(obs_exp).detach()
        loss, info = self.loss_fn(obs_z, exp_z, sim_is_window=False)
        if "per_step_costs" not in info:
            raise ValueError("BestOfK info missing per_step_costs for OTIL")
        per_step_costs = info.pop("per_step_costs")  # [B,T]
        per_step_rewards = self._build_pseudo_rewards(per_step_costs)
        info["pseudo_reward_mean"] = per_step_rewards.detach().mean()
        info["pseudo_reward_std"] = per_step_rewards.detach().std(unbiased=False)
        info["imitation_loss"] = loss.detach()

        step_rewards = per_step_rewards.transpose(0, 1)  # [T,B]
        step_cost = per_step_costs.transpose(0, 1)  # [T,B]

        with torch.no_grad():
            self.rew_buf.copy_(step_rewards.detach())
        # use live next_values for actor gradients; buffers stay detached for critic use
        next_vs_live = avg_next_values if self.actor_loss_avgcritics else next_values
        returns = self._compute_returns_from_rewards(step_cost, next_vs=next_vs_live)
        info["returns_mean"] = returns.detach().mean()

        actor_loss = -returns.mean()

        self.agent_steps += self.horizon_len * self.num_envs
        return actor_loss, info

    def _build_pseudo_rewards(self, per_step_costs: torch.Tensor) -> torch.Tensor:
        rewards_method = "exp"
        if rewards_method == "neg_loss":
            rewards = -per_step_costs
        elif rewards_method == "exp":
            rewards = torch.exp(-per_step_costs)
        elif rewards_method == "log_exp":
            rewards = -torch.log(1 - torch.exp(-per_step_costs) + 1e-10)
        else:
            raise NotImplementedError(rewards_method)
        if self.critic_reward_normalize:
            mean = rewards.mean(dim=1, keepdim=True)
            std = rewards.std(dim=1, keepdim=True, unbiased=False)
            rewards = (rewards - mean) / (std + 1e-6)
        rewards = self.reward_shaper(rewards)
        rewards = rewards * self.critic_reward_scale
        return rewards

    def _compute_returns_from_rewards(
        self,
        step_cost: torch.Tensor | None = None,
        next_vs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rewards = -step_cost

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
