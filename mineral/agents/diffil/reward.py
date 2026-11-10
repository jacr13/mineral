from copy import deepcopy

import torch

from ...common.demos import get_demos
from .best_of_k import (
    BestOfK,
    OTSinkhornCriterion,
    SequenceCosineCriterion,
    SequenceRegressionCriterion,
    SoftDTWCriterion,
)


def build_focus_loss(config, horizon_len, feature_dim, device):
    """Build a detached sequence objective with the same settings as FOCUS."""
    imitation_loss_type = config.get("imitation_loss_type", "ot")
    if imitation_loss_type == "ot":
        criterion = OTSinkhornCriterion(
            eps=float(config.get("loss_ot_eps", 0.1)),
            iters=int(config.get("loss_ot_iters", 60)),
            cost_type=config.get("loss_ot_cost_type", "l2"),
            huber_delta=float(config.get("loss_ot_huber_delta", 1.0)),
            use_huber_speedup=config.get("loss_use_huber_speedup", True),
        )
    elif imitation_loss_type in ("softdtw", "softdtw_subseq"):
        subseq = imitation_loss_type == "softdtw_subseq"
        criterion = SoftDTWCriterion(
            gamma=float(config.get("loss_dtw_gamma", 0.1)),
            cost_type=config.get("loss_ot_cost_type", "l2"),
            huber_delta=float(config.get("loss_ot_huber_delta", 1.0)),
            subsequence=subseq,
            open_end=subseq,
            normalize=config.get("loss_dtw_normalize", "rows"),
        )
    elif imitation_loss_type == "l2":
        criterion = SequenceRegressionCriterion(use_huber=False, reduction="mean")
    elif imitation_loss_type == "cosine":
        criterion = SequenceCosineCriterion()
    else:
        raise NotImplementedError(f"Unsupported FOCUS imitation loss: {imitation_loss_type}")

    input_type = config.get("input_type", "state")
    return BestOfK(
        T=horizon_len if input_type == "state" else horizon_len + 1,
        K=int(config.get("loss_best_of_k_k", 8)),
        tau=float(config.get("loss_best_of_k_tau", 0.5)),
        input_type=input_type,
        detach_prev_obs=config.get("loss_use_detached_prev_obs", True),
        mlp_features_dim=config.get("loss_mlp_features_dim", None),
        feature_dim=feature_dim,
        return_per_step_costs=True,
        criterion=criterion,
        device=device,
    )


class DetachedOTRewardMixin:
    """Common detached Best-of-K OT reward for zeroth-order RL agents.

    Subclasses provide ``_encode_ot_obs`` so the reward uses the same policy
    representation as the underlying PPO or SAC agent. Simulator trajectories,
    policy features, OT plans, costs, and rewards are all evaluated under
    ``torch.no_grad`` before being inserted into the RL buffer.
    """

    def _init_detached_ot_reward(self):
        self.focus_config = self.full_cfg.agent.get("focus", {})
        self.focus_input_type = self.focus_config.get("input_type", "state")
        if self.focus_input_type not in ("state", "state_state"):
            raise ValueError(f"Invalid FOCUS input_type: {self.focus_input_type}")

        # Attributes consumed by FOCUS._build_pseudo_rewards. Reusing that method
        # keeps the reward transformation byte-for-byte identical to FOCUS.
        self.critic_reward_mapping = self.focus_config.get("critic_reward_mapping", "log_exp")
        self.critic_reward_shapping = self.focus_config.get("critic_reward_shapping", True)
        self.critic_reward_normalize = self.focus_config.get("critic_reward_normalize", False)
        self.critic_reward_scale = self.focus_config.get("critic_reward_scale", 1.0)

        self.demos = get_demos(self.device, **self.focus_config.get("demos", {}))
        with torch.no_grad():
            demo_obs = self._normalize_ot_obs(self.demos["obs"], self.obs_rms)
            demo_z = self._encode_ot_obs(demo_obs)
        if not torch.is_tensor(demo_z):
            raise TypeError("FOCUS policy encoder must return a tensor")

        self.loss_fn = build_focus_loss(
            self.focus_config,
            horizon_len=self._ot_reward_horizon,
            feature_dim=demo_z.shape[-1],
            device=self.device,
        ).to(self.device)
        # The OT objective is a fixed reward function for these baselines.
        self.loss_fn.requires_grad_(False)
        self._last_ot_reward_stats = {}

    def _snapshot_ot_normalizer(self):
        return deepcopy(self.obs_rms) if self.obs_rms is not None else None

    @staticmethod
    def _normalize_ot_obs(obs, obs_rms):
        if obs_rms is None:
            return obs
        return {k: obs_rms[k].normalize(v) for k, v in obs.items()}

    def _new_ot_window(self, initial_obs):
        if self.focus_input_type == "state_state":
            return {k: [v.detach().clone()] for k, v in initial_obs.items()}
        return {k: [] for k in initial_obs}

    @staticmethod
    def _append_ot_window(obs_window, real_next_obs):
        for key, value in real_next_obs.items():
            obs_window[key].append(value.detach().clone())

    @torch.no_grad()
    def _compute_detached_ot_rewards(self, obs_window, obs_rms):
        obs_window = {k: torch.stack(v, dim=1) for k, v in obs_window.items()}
        sim_obs = self._normalize_ot_obs(obs_window, obs_rms)
        expert_obs = self._normalize_ot_obs(self.demos["obs"], obs_rms)
        sim_z = self._encode_ot_obs(sim_obs)
        expert_z = self._encode_ot_obs(expert_obs).detach()

        loss, info = self.loss_fn(sim_z, expert_z, sim_is_window=True)
        if "per_step_costs" not in info:
            raise ValueError("BestOfK info missing per_step_costs for detached FOCUS reward")
        costs = info["per_step_costs"].detach()
        # Import lazily to avoid a package import cycle during agent discovery.
        from .focus import FOCUS

        rewards = FOCUS._build_pseudo_rewards(self, costs).detach()
        self._last_ot_reward_stats = {
            "imitation_loss": loss.detach(),
            "cost_mean": costs.mean(),
            "cost_std": costs.std(unbiased=False),
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std(unbiased=False),
        }
        return rewards

    def _encode_ot_obs(self, obs):
        raise NotImplementedError
