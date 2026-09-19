#!/usr/bin/env python3
"""Paired test of whether Best-of-K rescues a phase-misaligned target.

For every query rollout and requested phase mismatch, K=1 receives one expert
segment at that mismatch. K>1 receives the exact same segment in slot zero,
plus K-1 uniformly sampled expert segments. This pairing isolates the benefit
of access to extra candidates: any improvement is caused by Best-of-K finding
a better alternative, not by an easier initial draw.

The script reuses the phase estimator, independent all-window oracle, and OT
criteria from ``phase_mismatch_diagnostic.py``. Without ``--rollout-path``,
held-out expert trajectories are used as a clean sanity check. For evidence
about policy guidance, pass rollouts collected from a learned policy.

Example:
    python scripts/paired_phase_rescue_diagnostic.py --env hopper --device cuda

    python scripts/paired_phase_rescue_diagnostic.py \
        --env hopper \
        --rollout-path workdir/hopper_agent_rollouts.pt \
        --num-query-windows 512 \
        --sampling-repeats 4 \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import phase_mismatch_diagnostic as diagnostic
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = diagnostic.build_parser()
    parser.description = (
        "Paired phase-rescue test: K=1 and Best-of-K share the same controlled "
        "mismatched candidate, while Best-of-K receives K-1 random alternatives."
    )
    parser.add_argument(
        "--sampling-repeats",
        type=int,
        default=1,
        help="Independent candidate pools per query window; increase for tighter estimates.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of trajectory-cluster bootstrap samples for 95%% confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=12345,
        help="Random seed for the trajectory-cluster bootstrap.",
    )
    return parser


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: torch.Tensor) -> float | None:
    return diagnostic.finite_mean(values)

def cluster_bootstrap_mean_ci(
    values: torch.Tensor,
    cluster_ids: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    num_samples: int = 5000,
    seed: int = 12345,
) -> tuple[float | None, float | None]:
    """Bootstrap a mean while resampling complete trajectory clusters.

    All repetitions and query windows belonging to the same trajectory remain
    together. For conditional metrics, `valid` identifies observations included
    in the denominator.
    """
    values = values.detach().float().cpu().reshape(-1)
    cluster_ids = cluster_ids.detach().cpu().reshape(-1)

    if valid is None:
        valid = torch.ones(values.shape, dtype=torch.bool)
    else:
        valid = valid.detach().bool().cpu().reshape(-1)

    finite = torch.isfinite(values)
    valid = valid & finite

    unique_clusters = torch.unique(cluster_ids, sorted=True)
    if unique_clusters.numel() < 2:
        return None, None

    numerators: list[torch.Tensor] = []
    denominators: list[torch.Tensor] = []

    for cluster_id in unique_clusters:
        mask = (cluster_ids == cluster_id) & valid
        numerators.append(values[mask].sum())
        denominators.append(mask.sum().float())

    cluster_numerators = torch.stack(numerators)
    cluster_denominators = torch.stack(denominators)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.randint(
        low=0,
        high=unique_clusters.numel(),
        size=(num_samples, unique_clusters.numel()),
        generator=generator,
    )

    bootstrap_numerators = cluster_numerators[draws].sum(dim=1)
    bootstrap_denominators = cluster_denominators[draws].sum(dim=1)

    usable = bootstrap_denominators > 0
    if not usable.any():
        return None, None

    estimates = (
        bootstrap_numerators[usable] / bootstrap_denominators[usable]
    )

    bounds = torch.quantile(
        estimates,
        torch.tensor([0.025, 0.975]),
    )
    return float(bounds[0]), float(bounds[1])

def prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    defaults = diagnostic.ENV_DEFAULTS[args.env]
    expert_path = diagnostic.resolve_path(args.expert_path or Path(defaults["expert_path"]))
    rollout_path = diagnostic.resolve_path(args.rollout_path) if args.rollout_path else None
    input_type = args.input_type or defaults["input_type"]
    period_min = args.period_min or defaults["period_min"]
    period_max = args.period_max or defaults["period_max"]
    device = diagnostic.choose_device(args.device)

    expert_observations, expert_lengths = diagnostic.load_trajectories(expert_path, args.obs_key)
    if expert_observations.shape[0] < args.reference_trajectories:
        raise ValueError(
            f"Requested {args.reference_trajectories} reference trajectories, "
            f"but {expert_path} contains {expert_observations.shape[0]}"
        )
    reference_observations = expert_observations[: args.reference_trajectories]
    reference_lengths = expert_lengths[: args.reference_trajectories]

    if rollout_path is None:
        query_start = args.reference_trajectories
        query_stop = query_start + args.query_trajectories
        if expert_observations.shape[0] < query_stop:
            raise ValueError(
                f"Held-out mode needs {query_stop} trajectories, but only "
                f"{expert_observations.shape[0]} are available"
            )
        query_observations = expert_observations[query_start:query_stop]
        query_lengths = expert_lengths[query_start:query_stop]
        query_source = "held_out_expert"
    else:
        rollout_observations, rollout_lengths = diagnostic.load_trajectories(rollout_path, args.obs_key)
        query_count = min(args.query_trajectories, rollout_observations.shape[0])
        query_observations = rollout_observations[:query_count]
        query_lengths = rollout_lengths[:query_count]
        query_source = "agent_rollout"

    if reference_observations.shape[-1] != query_observations.shape[-1]:
        raise ValueError(
            f"Reference feature dimension {reference_observations.shape[-1]} does not match "
            f"query dimension {query_observations.shape[-1]}"
        )

    mean, std = diagnostic.valid_statistics(reference_observations, reference_lengths)
    phase_dimensions = std > 1e-4
    if not phase_dimensions.any():
        raise ValueError("All reference observation dimensions are effectively constant")
    phase_reference = (reference_observations[..., phase_dimensions] - mean[phase_dimensions]) / std[
        phase_dimensions
    ].clamp_min(1e-6)

    if args.period is None:
        period, period_candidates = diagnostic.estimate_period(
            phase_reference,
            reference_lengths,
            period_min,
            period_max,
            args.period_burn_in,
        )
    else:
        period = args.period
        period_candidates = []

    template_trajectory = args.template_trajectory
    if not 0 <= template_trajectory < reference_observations.shape[0]:
        raise ValueError("--template-trajectory is outside the reference bank")
    template_length = int(reference_lengths[template_trajectory].item())
    template_start = min(args.template_start, template_length - period)
    if template_start < 0:
        raise ValueError("The selected template trajectory is shorter than the gait period")
    cycle = phase_reference[template_trajectory, template_start : template_start + period]
    probe_steps = min(args.phase_probe_steps, period, args.horizon)
    signatures = diagnostic.template_signatures(cycle, probe_steps)

    normalized_reference = diagnostic.normalize_observations(
        reference_observations,
        mean,
        std,
        args.normalization,
    )
    normalized_query = diagnostic.normalize_observations(
        query_observations,
        mean,
        std,
        args.normalization,
    )
    raw_window_size = args.horizon + 1 if input_type == "state_state" else args.horizon

    reference_raw, reference_trajectory_ids, reference_starts = diagnostic.extract_windows(
        normalized_reference,
        reference_lengths,
        raw_window_size,
        args.reference_stride,
    )
    reference_phase_raw, phase_trajectory_ids, phase_starts = diagnostic.extract_windows(
        phase_reference,
        reference_lengths,
        raw_window_size,
        args.reference_stride,
    )
    if not torch.equal(reference_trajectory_ids, phase_trajectory_ids) or not torch.equal(
        reference_starts, phase_starts
    ):
        raise RuntimeError("Internal reference-window metadata mismatch")

    query_raw, query_trajectory_ids, query_starts = diagnostic.extract_windows(
        normalized_query,
        query_lengths,
        raw_window_size,
        args.query_stride,
    )
    num_queries = min(args.num_query_windows, query_raw.shape[0])
    query_selection = torch.randperm(query_raw.shape[0], generator=args.generator)[:num_queries]
    query_raw = query_raw[query_selection]
    query_trajectory_ids = query_trajectory_ids[query_selection]
    query_starts = query_starts[query_selection]

    reference_features = diagnostic.make_features(reference_raw, input_type).to(device)
    query_features = diagnostic.make_features(query_raw, input_type).to(device)
    reference_phases, phase_assignment_costs = diagnostic.assign_phases(
        reference_phase_raw.to(device),
        signatures.to(device),
        probe_steps,
        args.oracle_batch_size,
    )
    reference_phases = reference_phases.cpu()

    oracle_steps = args.horizon if args.oracle_steps == 0 else min(args.oracle_steps, args.horizon)
    oracle_indices, oracle_costs = diagnostic.nearest_rows(
        query_features[:, :oracle_steps].reshape(query_features.shape[0], -1),
        reference_features[:, :oracle_steps].reshape(reference_features.shape[0], -1),
        args.oracle_batch_size,
    )
    oracle_indices = oracle_indices.cpu()
    oracle_costs = oracle_costs.cpu()
    oracle_phases = reference_phases[oracle_indices]

    return {
        "expert_path": expert_path,
        "rollout_path": rollout_path,
        "query_source": query_source,
        "input_type": input_type,
        "device": device,
        "period": period,
        "period_candidates": period_candidates,
        "template_trajectory": template_trajectory,
        "template_start": template_start,
        "probe_steps": probe_steps,
        "phase_dimensions": phase_dimensions,
        "phase_assignment_costs": phase_assignment_costs.cpu(),
        "reference_observations": reference_observations,
        "query_observations": query_observations,
        "reference_features": reference_features,
        "query_features": query_features,
        "reference_phases": reference_phases,
        "reference_trajectory_ids": reference_trajectory_ids,
        "reference_starts": reference_starts,
        "query_trajectory_ids": query_trajectory_ids,
        "query_starts": query_starts,
        "oracle_indices": oracle_indices,
        "oracle_costs": oracle_costs,
        "oracle_phases": oracle_phases,
        "oracle_steps": oracle_steps,
    }


def paired_summary(
    mismatch: float,
    candidate_indices: torch.Tensor,
    costs: torch.Tensor,
    data: dict[str, Any],
    args: argparse.Namespace,
    repeated_query_ids: torch.Tensor,
    repeat_ids: torch.Tensor,
    repeated_oracle_indices: torch.Tensor,
    repeated_oracle_costs: torch.Tensor,
    repeated_oracle_phases: torch.Tensor,
    oracle_criterion_costs: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    num_pairs, k = costs.shape
    rows = torch.arange(num_pairs)
    candidate_phases = data["reference_phases"][candidate_indices]
    phase_errors = diagnostic.circular_phase_error(
        candidate_phases,
        repeated_oracle_phases,
        data["period"],
    )
    close = phase_errors <= args.phase_tolerance
    weights = torch.softmax(-costs / args.tau, dim=1)
    best_k = costs.argmin(dim=1)
    selected_indices = candidate_indices[rows, best_k]

    k1_cost = costs[:, 0]
    k1_phase_error = phase_errors[:, 0]
    k1_close = close[:, 0]
    best_cost = costs[rows, best_k]
    best_phase_error = phase_errors[rows, best_k]
    best_close = close[rows, best_k]
    coverage = close.any(dim=1)
    extra_close_available = close[:, 1:].any(dim=1)
    rescue = (~k1_close) & best_close
    harm = k1_close & (~best_close)
    phase_improved = best_phase_error < k1_phase_error

    soft_weighted_cost = (weights * costs).sum(dim=1)
    soft_expected_phase_error = (weights * phase_errors).sum(dim=1)
    soft_close_mass = (weights * close.float()).sum(dim=1)
    softmin_objective = -args.tau * torch.logsumexp(-costs / args.tau, dim=1)
    if k == 1:
        normalized_entropy = torch.zeros(num_pairs)
        effective_candidates = torch.ones(num_pairs)
    else:
        raw_entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
        normalized_entropy = raw_entropy / math.log(k)
        effective_candidates = raw_entropy.exp()

    rescue_when_needed = rescue[~k1_close].float() if (~k1_close).any() else torch.tensor([])
    conditional_selection = best_close[coverage].float() if coverage.any() else torch.tensor([])
    hard_reduction = k1_cost - best_cost
    soft_reduction = k1_cost - soft_weighted_cost
    objective_reduction = k1_cost - softmin_objective
    phase_reduction = k1_phase_error - best_phase_error

    summary = {
        "target_mismatch": mismatch,
        "num_paired_samples": num_pairs,
        "k": k,
        "k1_close_fraction": finite_mean(k1_close.float()),
        "best_of_k_candidate_coverage_fraction": finite_mean(coverage.float()),
        "best_of_k_extra_close_available_fraction": finite_mean(extra_close_available.float()),
        "best_of_k_selected_close_fraction": finite_mean(best_close.float()),
        "best_of_k_conditional_selection_fraction": finite_mean(conditional_selection),
        "rescue_fraction_all": finite_mean(rescue.float()),
        "rescue_fraction_when_k1_wrong": finite_mean(rescue_when_needed),
        "harm_fraction_all": finite_mean(harm.float()),
        "best_of_k_selects_extra_candidate_fraction": finite_mean((best_k != 0).float()),
        "phase_improved_fraction": finite_mean(phase_improved.float()),
        "k1_phase_error_mean": finite_mean(k1_phase_error),
        "best_of_k_phase_error_mean": finite_mean(best_phase_error),
        "paired_phase_error_reduction_mean": finite_mean(phase_reduction),
        "soft_expected_phase_error_mean": finite_mean(soft_expected_phase_error),
        "soft_close_phase_weight_mass_mean": finite_mean(soft_close_mass),
        "soft_normalized_entropy_mean": finite_mean(normalized_entropy),
        "soft_effective_candidates_mean": finite_mean(effective_candidates),
        "soft_max_weight_mean": finite_mean(weights.max(dim=1).values),
        "oracle_criterion_cost_mean": finite_mean(oracle_criterion_costs),
        "k1_cost_mean": finite_mean(k1_cost),
        "best_of_k_hard_cost_mean": finite_mean(best_cost),
        "best_of_k_soft_weighted_cost_mean": finite_mean(soft_weighted_cost),
        "best_of_k_softmin_objective_mean": finite_mean(softmin_objective),
        "paired_hard_cost_reduction_mean": finite_mean(hard_reduction),
        "paired_soft_weighted_cost_reduction_mean": finite_mean(soft_reduction),
        "paired_softmin_objective_reduction_mean": finite_mean(objective_reduction),
        "oracle_aligned_l2_mean": finite_mean(repeated_oracle_costs),
    }

    # Query windows and sampling repeats from one trajectory are correlated.
    repeated_trajectory_ids = data["query_trajectory_ids"][repeated_query_ids]

    bootstrap_metrics = {
        "rescue_fraction_all": (rescue.float(), None),
        "k1_phase_error_mean": (k1_phase_error, None),
        "best_of_k_phase_error_mean": (best_phase_error, None),
        "best_of_k_candidate_coverage_fraction": (
            coverage.float(),
            None,
        ),
        "best_of_k_selected_close_fraction": (
            best_close.float(),
            None,
        ),
        "best_of_k_conditional_selection_fraction": (
            best_close.float(),
            coverage,
        ),
        "rescue_fraction_when_k1_wrong": (
            rescue.float(),
            ~k1_close,
        ),
        "harm_fraction_all": (
            harm.float(),
            None,
        ),
        "phase_improved_fraction": (
            phase_improved.float(),
            None,
        ),
        "paired_phase_error_reduction_mean": (
            phase_reduction,
            None,
        ),
        "paired_hard_cost_reduction_mean": (
            hard_reduction,
            None,
        ),
        "paired_soft_weighted_cost_reduction_mean": (
            soft_reduction,
            None,
        ),
    }

    for metric_name, (metric_values, valid_mask) in bootstrap_metrics.items():
        ci_low, ci_high = cluster_bootstrap_mean_ci(
            metric_values,
            repeated_trajectory_ids,
            valid=valid_mask,
            num_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        summary[f"{metric_name}_ci95_low"] = ci_low
        summary[f"{metric_name}_ci95_high"] = ci_high

    details = []
    reference_trajectory_ids = data["reference_trajectory_ids"]
    reference_starts = data["reference_starts"]
    for pair_id in range(num_pairs):
        query_id = int(repeated_query_ids[pair_id].item())
        oracle_index = int(repeated_oracle_indices[pair_id].item())
        base_index = int(candidate_indices[pair_id, 0].item())
        selected_index = int(selected_indices[pair_id].item())
        candidate_list = candidate_indices[pair_id]
        details.append(
            {
                "target_mismatch": mismatch,
                "pair_id": pair_id,
                "query_id": query_id,
                "sampling_repeat": int(repeat_ids[pair_id].item()),
                "query_trajectory": int(data["query_trajectory_ids"][query_id].item()),
                "query_start": int(data["query_starts"][query_id].item()),
                "oracle_reference_trajectory": int(reference_trajectory_ids[oracle_index].item()),
                "oracle_reference_start": int(reference_starts[oracle_index].item()),
                "oracle_phase": int(repeated_oracle_phases[pair_id].item()),
                "base_reference_trajectory": int(reference_trajectory_ids[base_index].item()),
                "base_reference_start": int(reference_starts[base_index].item()),
                "base_phase": int(candidate_phases[pair_id, 0].item()),
                "base_phase_error": float(k1_phase_error[pair_id].item()),
                "base_close": bool(k1_close[pair_id].item()),
                "base_cost": float(k1_cost[pair_id].item()),
                "candidate_reference_trajectories": json.dumps(reference_trajectory_ids[candidate_list].tolist()),
                "candidate_reference_starts": json.dumps(reference_starts[candidate_list].tolist()),
                "candidate_phases": json.dumps(candidate_phases[pair_id].tolist()),
                "candidate_phase_errors": json.dumps(phase_errors[pair_id].tolist()),
                "candidate_costs": json.dumps(costs[pair_id].tolist()),
                "softmin_weights": json.dumps(weights[pair_id].tolist()),
                "close_candidate_available": bool(coverage[pair_id].item()),
                "extra_close_candidate_available": bool(extra_close_available[pair_id].item()),
                "selected_slot": int(best_k[pair_id].item()),
                "selected_reference_trajectory": int(reference_trajectory_ids[selected_index].item()),
                "selected_reference_start": int(reference_starts[selected_index].item()),
                "selected_phase": int(data["reference_phases"][selected_index].item()),
                "selected_phase_error": float(best_phase_error[pair_id].item()),
                "selected_close": bool(best_close[pair_id].item()),
                "rescued": bool(rescue[pair_id].item()),
                "harmed": bool(harm[pair_id].item()),
                "phase_error_reduction": float(phase_reduction[pair_id].item()),
                "hard_cost_reduction": float(hard_reduction[pair_id].item()),
                "soft_weighted_cost_reduction": float(soft_reduction[pair_id].item()),
                "soft_close_phase_weight_mass": float(soft_close_mass[pair_id].item()),
                "soft_normalized_entropy": float(normalized_entropy[pair_id].item()),
            }
        )
    return summary, details


def plot_results(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping the PNG plot")
        return

    summary_rows.sort(key=lambda row: row["target_mismatch"])
    x = [row["target_mismatch"] for row in summary_rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(x, [row["k1_close_fraction"] for row in summary_rows], marker="o", label="paired K=1")
    axes[0, 0].plot(
        x,
        [row["best_of_k_selected_close_fraction"] for row in summary_rows],
        marker="o",
        label="Best-of-K selected",
    )
    axes[0, 0].plot(
        x,
        [row["best_of_k_candidate_coverage_fraction"] for row in summary_rows],
        marker="o",
        linestyle="--",
        label="Best-of-K coverage",
    )
    axes[0, 0].set_ylabel("Close-phase fraction")
    axes[0, 0].legend()

    axes[0, 1].plot(x, [row["k1_phase_error_mean"] for row in summary_rows], marker="o", label="paired K=1")
    axes[0, 1].plot(
        x,
        [row["best_of_k_phase_error_mean"] for row in summary_rows],
        marker="o",
        label="Best-of-K",
    )
    axes[0, 1].plot(x, x, color="grey", linestyle="--", label="requested mismatch")
    axes[0, 1].set_ylabel("Selected circular phase error")
    axes[0, 1].legend()

    axes[1, 0].plot(x, [row["k1_cost_mean"] for row in summary_rows], marker="o", label="paired K=1")
    axes[1, 0].plot(
        x,
        [row["best_of_k_hard_cost_mean"] for row in summary_rows],
        marker="o",
        label="Best-of-K hard",
    )
    axes[1, 0].plot(
        x,
        [row["best_of_k_soft_weighted_cost_mean"] for row in summary_rows],
        marker="o",
        label="Best-of-K soft weighted",
    )
    axes[1, 0].set_ylabel("Criterion cost")
    axes[1, 0].legend()

    axes[1, 1].plot(
        x,
        [row["rescue_fraction_when_k1_wrong"] or 0.0 for row in summary_rows],
        marker="o",
        label="rescue when K=1 wrong",
    )
    axes[1, 1].plot(
        x,
        [row["soft_close_phase_weight_mass_mean"] for row in summary_rows],
        marker="o",
        label="soft close-phase mass",
    )
    axes[1, 1].plot(
        x,
        [row["soft_normalized_entropy_mean"] for row in summary_rows],
        marker="o",
        label="soft entropy",
    )
    axes[1, 1].set_ylabel("Rescue / weight statistic")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Controlled mismatch of shared K=1 candidate (cycles)")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    diagnostic.validate_args(args)
    if args.k < 2:
        raise ValueError("The paired rescue test requires -k >= 2")
    if args.sampling_repeats <= 0:
        raise ValueError("--sampling-repeats must be positive")

    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    torch.manual_seed(args.seed)
    args.generator = torch.Generator(device="cpu").manual_seed(args.seed)
    data = prepare_data(args)
    output_dir = diagnostic.resolve_path(
        args.output_dir or Path("workdir") / "paired_phase_rescue_diagnostic" / args.env
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    num_queries = data["query_features"].shape[0]
    repeated_query_ids = torch.arange(num_queries).repeat_interleave(args.sampling_repeats)
    repeat_ids = torch.arange(args.sampling_repeats).repeat(num_queries)
    repeated_queries = data["query_features"].repeat_interleave(args.sampling_repeats, dim=0)
    repeated_oracle_indices = data["oracle_indices"].repeat_interleave(args.sampling_repeats)
    repeated_oracle_costs = data["oracle_costs"].repeat_interleave(args.sampling_repeats)
    repeated_oracle_phases = data["oracle_phases"].repeat_interleave(args.sampling_repeats)

    criterion = diagnostic.build_criterion(args)
    oracle_criterion_costs = diagnostic.candidate_costs(
        repeated_queries,
        data["reference_features"],
        repeated_oracle_indices.unsqueeze(1),
        criterion,
        args.criterion_batch_size,
    )[:, 0]

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for mismatch in args.mismatch_offsets:
        base_candidate = diagnostic.sample_controlled_candidates(
            repeated_oracle_phases,
            data["reference_phases"],
            data["period"],
            mismatch,
            args.mismatch_width,
            1,
            args.generator,
        )
        extras = torch.randint(
            data["reference_features"].shape[0],
            (repeated_queries.shape[0], args.k - 1),
            generator=args.generator,
        )
        candidates = torch.cat([base_candidate, extras], dim=1)
        if not torch.equal(candidates[:, :1], base_candidate):
            raise RuntimeError("Pairing invariant failed: Best-of-K did not preserve the K=1 candidate")
        costs = diagnostic.candidate_costs(
            repeated_queries,
            data["reference_features"],
            candidates,
            criterion,
            args.criterion_batch_size,
        )
        summary, details = paired_summary(
            mismatch,
            candidates,
            costs,
            data,
            args,
            repeated_query_ids,
            repeat_ids,
            repeated_oracle_indices,
            repeated_oracle_costs,
            repeated_oracle_phases,
            oracle_criterion_costs,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)

    metadata = {
        "environment": args.env,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items() if key != "generator"},
        "torch_version": torch.__version__,
        "phase_reference_scope": "one shared template per environment, not per demonstration",
        "expert_path": str(data["expert_path"]),
        "rollout_path": str(data["rollout_path"]) if data["rollout_path"] else None,
        "query_source": data["query_source"],
        "input_type": data["input_type"],
        "criterion": args.criterion,
        "ot_cost": args.ot_cost if args.criterion == "ot" else None,
        "normalization": args.normalization,
        "observation_dim": data["reference_observations"].shape[-1],
        "phase_dimensions": int(data["phase_dimensions"].sum().item()),
        "reference_trajectories": data["reference_observations"].shape[0],
        "query_trajectories": data["query_observations"].shape[0],
        "reference_windows": data["reference_features"].shape[0],
        "query_windows": num_queries,
        "sampling_repeats": args.sampling_repeats,
        "paired_samples_per_mismatch": repeated_queries.shape[0],
        "horizon": args.horizon,
        "oracle_steps": data["oracle_steps"],
        "estimated_period": data["period"],
        "period_local_minima": data["period_candidates"],
        "template_trajectory": data["template_trajectory"],
        "template_start": data["template_start"],
        "phase_probe_steps": data["probe_steps"],
        "phase_assignment_l2_mean": finite_mean(data["phase_assignment_costs"]),
        "phase_tolerance": args.phase_tolerance,
        "k": args.k,
        "tau": args.tau,
        "seed": args.seed,
        "device": str(data["device"]),
        "pairing": (
            "For each query and mismatch, K=1 uses candidate slot 0. Best-of-K uses the identical "
            "slot-0 candidate plus K-1 uniformly random reference windows."
        ),
        "interpretation": (
            "Rescue means K=1's shared candidate is outside the close-phase tolerance while the "
            "lowest-cost candidate in the paired Best-of-K pool is inside it. This isolates candidate-set "
            "benefit but does not by itself measure policy return or gradient alignment."
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "query_trajectory",
        "confidence_interval": "trajectory-cluster bootstrap percentile 95% CI",
    }

    write_csv(output_dir / "paired_summary.csv", summary_rows)
    write_csv(output_dir / "paired_per_query.csv", detail_rows)
    with (output_dir / "paired_results.json").open("w") as file:
        json.dump({"metadata": metadata, "results": summary_rows}, file, indent=2)
    plot_results(output_dir / "paired_phase_rescue.png", summary_rows)

    print(f"Environment: {args.env}")
    print(f"Query source: {data['query_source']}")
    print(f"Estimated gait period: {data['period']} steps")
    print(f"Reference/query windows: {data['reference_features'].shape[0]}/{num_queries}")
    print(f"Paired samples per mismatch: {repeated_queries.shape[0]}")
    print()
    header = (
        f"{'mismatch':>9} {'K1 close':>9} {'K close':>9} {'coverage':>9} "
        f"{'rescue':>9} {'harm':>9} {'K1 phase':>9} {'K phase':>9} {'cost gain':>10} {'soft mass':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        rescue = row["rescue_fraction_when_k1_wrong"]
        rescue_text = "n/a" if rescue is None else f"{rescue:.3f}"
        print(
            f"{row['target_mismatch']:>9.3f} "
            f"{row['k1_close_fraction']:>9.3f} "
            f"{row['best_of_k_selected_close_fraction']:>9.3f} "
            f"{row['best_of_k_candidate_coverage_fraction']:>9.3f} "
            f"{rescue_text:>9} "
            f"{row['harm_fraction_all']:>9.3f} "
            f"{row['k1_phase_error_mean']:>9.3f} "
            f"{row['best_of_k_phase_error_mean']:>9.3f} "
            f"{row['paired_hard_cost_reduction_mean']:>10.4f} "
            f"{row['soft_close_phase_weight_mass_mean']:>10.3f}"
        )
    print(f"\nWrote paired diagnostic outputs to {output_dir}")


if __name__ == "__main__":
    main()
