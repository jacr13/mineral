from copy import deepcopy

import torch

from ...common.demos import get_demos
from .best_of_k import (
    BestOfK,
    ORCACriterion,
    OTSinkhornCriterion,
    SequenceCosineCriterion,
    SequenceRegressionCriterion,
    SoftDTWCriterion,
)


def build_focus_loss(config, horizon_len, feature_dim, device):
    """The FOCUS sequence objective (criterion inside best-of-K) from a ``focus`` config.

    The one builder of FOCUS (first order) and of its zeroth-order variants (FOCUS-PPO, FOCUS-SAC), so that the
    objective cannot drift between them. Keys and defaults are those FOCUS has always read (defaults = the values of
    the reported runs). ``imitation_loss_type: index`` (no alignment) is step t against step t of the candidate with
    the configured ground cost.
    """
    get = config.get
    loss_type = get("imitation_loss_type", "ot")
    cost_type = get("loss_ot_cost_type", "l2")
    if loss_type == "index":
        loss_type = "cosine" if cost_type == "cosine" else "l2"
    huber_delta = float(get("loss_ot_huber_delta", 1.0))
    step_cost = get("loss_cost_units", "legacy")
    if loss_type == "ot":
        rho = get("loss_ot_marginal_rho", None)
        criterion = OTSinkhornCriterion(
            eps=float(get("loss_ot_eps", 0.1)),
            iters=int(get("loss_ot_iters", 60)),
            cost_type=cost_type,
            huber_delta=huber_delta,
            use_huber_speedup=get("loss_use_huber_speedup", True),
            detach_plan=bool(get("loss_ot_detach_plan", True)),
            marginal_rho=None if rho in (None, "null", "none") else float(rho),
            marginal_side=get("loss_ot_marginal_side", "both"),
            step_cost=step_cost,
        )
    elif loss_type in ("softdtw", "softdtw_subseq"):
        subseq = loss_type == "softdtw_subseq"
        criterion = SoftDTWCriterion(
            gamma=float(get("loss_dtw_gamma", 0.1)),
            cost_type=cost_type,
            huber_delta=huber_delta,
            subsequence=subseq,
            open_end=subseq,
            normalize=get("loss_dtw_normalize", "rows"),
            step_cost=step_cost,
        )
    elif loss_type == "orca":
        criterion = ORCACriterion(
            lam=float(get("loss_orca_lam", 0.25)),
            cost_type=cost_type,
            huber_delta=huber_delta,
            normalize_p=bool(get("loss_orca_normalize_p", False)),
            cost_mode=get("loss_orca_cost_mode", "neg_log_r"),
            normalize_cost=bool(get("loss_orca_normalize_cost", True)),
        )
    elif loss_type == "l2":
        criterion = SequenceRegressionCriterion(use_huber=False, reduction="mean")
    elif loss_type == "cosine":
        criterion = SequenceCosineCriterion()
    else:
        raise NotImplementedError(f"Unsupported FOCUS imitation loss: {loss_type}")

    input_type = get("input_type", "state")
    return BestOfK(
        T=horizon_len if input_type == "state" else horizon_len + 1,
        K=get("loss_best_of_k_k", 8),
        tau=float(get("loss_best_of_k_tau", 0.5)),
        input_type=input_type,
        detach_prev_obs=get("loss_use_detached_prev_obs", True),
        mlp_features_dim=get("loss_mlp_features_dim", None),
        feature_dim=feature_dim,
        return_per_step_costs=True,
        device=device,
        relative=bool(get("loss_best_of_k_relative", False)),
        criterion=criterion,
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
        self.critic_reward_minus_cost = self.focus_config.get("critic_reward_minus_cost", None)

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
