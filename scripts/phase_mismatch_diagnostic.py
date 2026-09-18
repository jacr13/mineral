#!/usr/bin/env python3
"""Offline diagnostic for Best-of-K temporal/phase matching.

The script compares random K=1, softmin Best-of-K, and hard-min Best-of-K
using identical candidate pools. It also constructs candidate pools at
controlled circular phase offsets from an independently computed oracle.

By default, the first expert trajectories form the reference bank and the
next trajectories act as held-out rollouts. Pass ``--rollout-path`` to use
states collected from a trained policy instead. Input files use Mineral's
demonstration format: a torch-saved dictionary whose ``obs`` value is either
a [trajectories, time, features] tensor or a dictionary containing one.

Example:
    python scripts/phase_mismatch_diagnostic.py --env hopper

    python scripts/phase_mismatch_diagnostic.py \
        --env ant \
        --rollout-path workdir/ant_eval_rollouts.pt \
        --num-query-windows 512 \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mineral.agents.otil.best_of_k import (  # noqa: E402, I001
    OTSinkhornCriterion,
    SequenceCosineCriterion,
    SequenceRegressionCriterion,
)



ENV_DEFAULTS = {
    "hopper": {
        "expert_path": "experts/demos/DFlex_hopper_demos128_return4812_len1000.pt",
        "input_type": "state",
        "period_min": 10,
        "period_max": 90,
    },
    "ant": {
        "expert_path": "experts/demos/DFlex_ant_demos128_return9318_len1000.pt",
        "input_type": "state_state",
        "period_min": 8,
        "period_max": 80,
    },
    "humanoid": {
        "expert_path": "experts/demos/DFlex_humanoid_demos128_return8116_len984.pt",
        "input_type": "state",
        "period_min": 8,
        "period_max": 120,
    },
    "snu_humanoid": {
        "expert_path": "experts/demos/DFlex_snu_demos128_return6246_len929.pt",
        "input_type": "state",
        "period_min": 8,
        "period_max": 120,
    },
}


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number")
    if any(item < 0.0 or item > 0.5 for item in values):
        raise argparse.ArgumentTypeError("Phase mismatch values must lie in [0, 0.5]")
    return values


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def torch_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(payload).__name__}")
    return payload


def select_observation(payload: dict[str, Any], obs_key: str | None, path: Path) -> torch.Tensor:
    if "obs" not in payload:
        raise KeyError(f"{path} does not contain an 'obs' entry")
    observations = payload["obs"]
    if isinstance(observations, dict):
        if obs_key is not None:
            if obs_key not in observations:
                raise KeyError(f"Observation key {obs_key!r} is not present; choices: {sorted(observations)}")
            observations = observations[obs_key]
        elif "obs" in observations:
            observations = observations["obs"]
        elif len(observations) == 1:
            observations = next(iter(observations.values()))
        else:
            raise ValueError(f"{path} has multiple observation keys; select one with --obs-key")
    if not torch.is_tensor(observations) or observations.ndim != 3:
        shape = getattr(observations, "shape", None)
        raise ValueError(f"Expected observations with shape [trajectories, time, features], got {shape}")
    return observations.detach().to(dtype=torch.float32, device="cpu")


def infer_lengths(payload: dict[str, Any], observations: torch.Tensor) -> torch.Tensor:
    num_trajectories, max_length = observations.shape[:2]
    lengths = payload.get("lengths")
    if torch.is_tensor(lengths):
        lengths = lengths[:num_trajectories].reshape(-1).to(dtype=torch.long, device="cpu")
    else:
        done = payload.get("done")
        lengths = torch.full((num_trajectories,), max_length, dtype=torch.long)
        if torch.is_tensor(done) and done.ndim >= 2:
            done = done[:num_trajectories].reshape(num_trajectories, max_length, -1).any(dim=-1)
            for trajectory_id in range(num_trajectories):
                terminal = torch.where(done[trajectory_id])[0]
                if terminal.numel() > 0:
                    lengths[trajectory_id] = int(terminal[0].item()) + 1
    return lengths.clamp(min=1, max=max_length)


def load_trajectories(path: Path, obs_key: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch_load(path)
    observations = select_observation(payload, obs_key, path)
    lengths = infer_lengths(payload, observations)
    if not torch.isfinite(observations).all():
        raise ValueError(f"Non-finite observations found in {path}")
    return observations, lengths


def valid_statistics(observations: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.cat(
        [observations[index, : int(length.item())] for index, length in enumerate(lengths)],
        dim=0,
    )
    mean = valid.mean(dim=0)
    std = valid.std(dim=0, unbiased=False)
    return mean, std


def normalize_observations(
    observations: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "none":
        return observations
    if mode == "zscore":
        return (observations - mean) / std.clamp_min(1e-6)
    raise ValueError(f"Unknown normalization mode: {mode}")


def estimate_period(
    observations: torch.Tensor,
    lengths: torch.Tensor,
    period_min: int,
    period_max: int,
    burn_in: int,
) -> tuple[int, list[tuple[int, float]]]:
    scores: list[tuple[int, float]] = []
    for lag in range(period_min, period_max + 1):
        per_trajectory = []
        for trajectory_id, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            start = min(burn_in, max(0, length - lag - 2))
            if length - start <= lag:
                continue
            current = observations[trajectory_id, start : length - lag]
            shifted = observations[trajectory_id, start + lag : length]
            per_trajectory.append((current - shifted).square().mean())
        if per_trajectory:
            score = torch.stack(per_trajectory).mean().item()
            scores.append((lag, score))
    if len(scores) < 3:
        raise ValueError("Not enough trajectory data to estimate a period")

    local_minima = [
        item
        for index, item in enumerate(scores[1:-1], start=1)
        if item[1] < scores[index - 1][1] and item[1] < scores[index + 1][1]
    ]
    candidates = local_minima or scores
    period = min(candidates, key=lambda item: item[1])[0]
    return period, sorted(local_minima, key=lambda item: item[1])[:10]


def extract_windows(
    observations: torch.Tensor,
    lengths: torch.Tensor,
    window_size: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    windows = []
    trajectory_ids = []
    starts = []
    for trajectory_id, length_tensor in enumerate(lengths):
        length = int(length_tensor.item())
        if length < window_size:
            continue
        trajectory = observations[trajectory_id, :length]
        trajectory_windows = trajectory.unfold(0, window_size, stride).movedim(-1, -2).contiguous()
        num_windows = trajectory_windows.shape[0]
        windows.append(trajectory_windows)
        trajectory_ids.append(torch.full((num_windows,), trajectory_id, dtype=torch.long))
        starts.append(torch.arange(num_windows, dtype=torch.long) * stride)
    if not windows:
        raise ValueError(f"No trajectory is long enough for a window of {window_size} steps")
    return torch.cat(windows), torch.cat(trajectory_ids), torch.cat(starts)


def make_features(raw_windows: torch.Tensor, input_type: str) -> torch.Tensor:
    if input_type == "state":
        return raw_windows
    if input_type == "state_state":
        return torch.cat([raw_windows[:, :-1], raw_windows[:, 1:]], dim=-1)
    raise ValueError(f"Unknown input type: {input_type}")


def template_signatures(cycle: torch.Tensor, probe_steps: int) -> torch.Tensor:
    period = cycle.shape[0]
    offsets = torch.arange(probe_steps).view(1, probe_steps)
    phases = torch.arange(period).view(period, 1)
    indices = (phases + offsets) % period
    return cycle[indices].reshape(period, -1)


@torch.no_grad()
def nearest_rows(
    queries: torch.Tensor,
    references: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_norm = references.square().sum(dim=1).view(1, -1)
    best_indices = []
    best_costs = []
    for start in range(0, queries.shape[0], batch_size):
        query = queries[start : start + batch_size]
        distances = query.square().sum(dim=1, keepdim=True) + reference_norm - 2.0 * query @ references.T
        distances.clamp_min_(0.0)
        values, indices = distances.min(dim=1)
        best_indices.append(indices)
        best_costs.append(values / queries.shape[1])
    return torch.cat(best_indices), torch.cat(best_costs)


@torch.no_grad()
def assign_phases(
    phase_windows: torch.Tensor,
    signatures: torch.Tensor,
    probe_steps: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probes = phase_windows[:, :probe_steps].reshape(phase_windows.shape[0], -1)
    return nearest_rows(probes, signatures, batch_size)


def build_criterion(args: argparse.Namespace):
    if args.criterion == "ot":
        return OTSinkhornCriterion(
            cost_type=args.ot_cost,
            use_huber_speedup=False,
        )
    if args.criterion == "l2":
        return SequenceRegressionCriterion(use_huber=False, reduction="mean")
    if args.criterion == "cosine":
        return SequenceCosineCriterion()
    raise ValueError(f"Unknown criterion: {args.criterion}")


@torch.no_grad()
def candidate_costs(
    queries: torch.Tensor,
    references: torch.Tensor,
    candidate_indices: torch.Tensor,
    criterion,
    batch_size: int,
) -> torch.Tensor:
    costs = []
    for start in range(0, queries.shape[0], batch_size):
        stop = start + batch_size
        query = queries[start:stop]
        indices = candidate_indices[start:stop].to(device=references.device)
        candidates = references[indices]
        criterion_output = criterion(query, candidates)
        batch_costs = criterion_output[0] if isinstance(criterion_output, tuple) else criterion_output
        costs.append(batch_costs.detach().cpu())
    return torch.cat(costs)


def circular_phase_error(candidate_phases: torch.Tensor, oracle_phases: torch.Tensor, period: int) -> torch.Tensor:
    difference = (candidate_phases - oracle_phases.unsqueeze(1)).abs()
    return torch.minimum(difference, period - difference).to(dtype=torch.float32) / period


def finite_mean(values: torch.Tensor) -> float | None:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return float(values.mean().item())


def summarize_pool(
    pool_name: str,
    target_mismatch: float | None,
    candidate_indices: torch.Tensor,
    costs: torch.Tensor,
    oracle_indices: torch.Tensor,
    oracle_costs: torch.Tensor,
    oracle_phases: torch.Tensor,
    reference_phases: torch.Tensor,
    reference_trajectory_ids: torch.Tensor,
    reference_starts: torch.Tensor,
    period: int,
    tau: float,
    phase_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    num_queries, k = costs.shape
    candidate_phases = reference_phases[candidate_indices]
    phase_errors = circular_phase_error(candidate_phases, oracle_phases, period)
    weights = torch.softmax(-costs / tau, dim=1)
    best_k = costs.argmin(dim=1)
    batch_indices = torch.arange(num_queries)
    selected_indices = candidate_indices[batch_indices, best_k]
    selected_errors = phase_errors[batch_indices, best_k]
    close = phase_errors <= phase_tolerance
    coverage = close.any(dim=1)
    selected_close = selected_errors <= phase_tolerance

    if k == 1:
        normalized_entropy = torch.zeros(num_queries)
    else:
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
        normalized_entropy = entropy / math.log(k)

    hard_cost = costs[batch_indices, best_k]
    soft_weighted_cost = (weights * costs).sum(dim=1)
    softmin_loss = -tau * torch.logsumexp(-costs / tau, dim=1)
    conditional = selected_close[coverage].float() if coverage.any() else torch.tensor([])

    summary = {
        "pool": pool_name,
        "target_mismatch": target_mismatch,
        "num_queries": num_queries,
        "k": k,
        "oracle_aligned_l2_mean": finite_mean(oracle_costs),
        "candidate_phase_error_mean": finite_mean(phase_errors),
        "candidate_min_phase_error_mean": finite_mean(phase_errors.min(dim=1).values),
        "candidate_coverage_fraction": finite_mean(coverage.float()),
        "selected_close_fraction": finite_mean(selected_close.float()),
        "conditional_selected_close_fraction": finite_mean(conditional),
        "highest_weight_phase_error_mean": finite_mean(selected_errors),
        "soft_expected_phase_error_mean": finite_mean((weights * phase_errors).sum(dim=1)),
        "soft_oracle_weight_mass_mean": finite_mean((weights * close.float()).sum(dim=1)),
        "soft_normalized_entropy_mean": finite_mean(normalized_entropy),
        "soft_max_weight_mean": finite_mean(weights.max(dim=1).values),
        "hard_cost_mean": finite_mean(hard_cost),
        "soft_weighted_cost_mean": finite_mean(soft_weighted_cost),
        "softmin_objective_mean": finite_mean(softmin_loss),
        "hard_exp_neg_cost_proxy_mean": finite_mean(torch.exp(-hard_cost)),
        "soft_exp_neg_cost_proxy_mean": finite_mean(torch.exp(-soft_weighted_cost)),
    }

    details = []
    for query_id in range(num_queries):
        oracle_index = int(oracle_indices[query_id].item())
        selected_index = int(selected_indices[query_id].item())
        candidate_list = candidate_indices[query_id]
        details.append(
            {
                "pool": pool_name,
                "target_mismatch": target_mismatch,
                "query_id": query_id,
                "oracle_reference_trajectory": int(reference_trajectory_ids[oracle_index].item()),
                "oracle_reference_start": int(reference_starts[oracle_index].item()),
                "oracle_phase": int(oracle_phases[query_id].item()),
                "oracle_aligned_l2": float(oracle_costs[query_id].item()),
                "candidate_reference_trajectories": json.dumps(reference_trajectory_ids[candidate_list].tolist()),
                "candidate_reference_starts": json.dumps(reference_starts[candidate_list].tolist()),
                "candidate_phases": json.dumps(candidate_phases[query_id].tolist()),
                "candidate_phase_errors": json.dumps(phase_errors[query_id].tolist()),
                "candidate_costs": json.dumps(costs[query_id].tolist()),
                "softmin_weights": json.dumps(weights[query_id].tolist()),
                "candidate_contains_close_phase": bool(coverage[query_id].item()),
                "selected_reference_trajectory": int(reference_trajectory_ids[selected_index].item()),
                "selected_reference_start": int(reference_starts[selected_index].item()),
                "selected_phase": int(reference_phases[selected_index].item()),
                "selected_phase_error": float(selected_errors[query_id].item()),
                "selected_close": bool(selected_close[query_id].item()),
                "soft_normalized_entropy": float(normalized_entropy[query_id].item()),
                "soft_max_weight": float(weights[query_id].max().item()),
                "hard_cost": float(hard_cost[query_id].item()),
                "soft_weighted_cost": float(soft_weighted_cost[query_id].item()),
            }
        )
    return summary, details


def sample_controlled_candidates(
    oracle_phases: torch.Tensor,
    reference_phases: torch.Tensor,
    period: int,
    target_mismatch: float,
    width: float,
    k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    phase_to_indices = [torch.where(reference_phases == phase)[0] for phase in range(period)]
    candidate_rows = []
    for oracle_phase_tensor in oracle_phases:
        oracle_phase = int(oracle_phase_tensor.item())
        allowed = []
        for phase in range(period):
            difference = abs(phase - oracle_phase)
            phase_error = min(difference, period - difference) / period
            if abs(phase_error - target_mismatch) <= width:
                allowed.append(phase_to_indices[phase])
        nonempty = [indices for indices in allowed if indices.numel() > 0]
        pool = torch.cat(nonempty) if nonempty else torch.tensor([], dtype=torch.long)
        if pool.numel() == 0:
            raise ValueError(
                f"No expert windows available near mismatch={target_mismatch}; "
                "increase --mismatch-width"
            )
        sampled = torch.randint(pool.numel(), (k,), generator=generator)
        candidate_rows.append(pool[sampled])
    return torch.stack(candidate_rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_mismatch_curve(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    controlled = [row for row in summary_rows if row["target_mismatch"] is not None]
    if not controlled:
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping the PNG plot")
        return

    controlled.sort(key=lambda row: row["target_mismatch"])
    x = [row["target_mismatch"] for row in controlled]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    axes[0].plot(x, [row["hard_cost_mean"] for row in controlled], marker="o", label="hard min")
    axes[0].plot(x, [row["soft_weighted_cost_mean"] for row in controlled], marker="o", label="soft weighted")
    axes[0].set_ylabel("Candidate cost")
    axes[0].legend()

    axes[1].plot(x, [row["highest_weight_phase_error_mean"] for row in controlled], marker="o")
    axes[1].plot(x, x, linestyle="--", color="grey", label="requested mismatch")
    axes[1].set_ylabel("Selected circular phase error")
    axes[1].legend()

    axes[2].plot(x, [row["soft_normalized_entropy_mean"] for row in controlled], marker="o", label="entropy")
    axes[2].plot(x, [row["soft_max_weight_mean"] for row in controlled], marker="o", label="max weight")
    axes[2].set_ylabel("Softmin concentration")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("Controlled initial phase mismatch (cycles)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure whether Best-of-K selects expert segments near an independent oracle gait phase."
    )
    parser.add_argument("--env", choices=sorted(ENV_DEFAULTS), default="hopper")
    parser.add_argument("--expert-path", type=Path, help="Reference expert trajectories (.pt).")
    parser.add_argument("--rollout-path", type=Path, help="Optional agent rollouts in Mineral demo format.")
    parser.add_argument("--obs-key", help="Observation dictionary key; defaults to 'obs' or the only key.")
    parser.add_argument("--reference-trajectories", type=int, default=8)
    parser.add_argument("--query-trajectories", type=int, default=8)
    parser.add_argument("--num-query-windows", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=32, help="Number of feature/cost timesteps.")
    parser.add_argument("--input-type", choices=("state", "state_state"))
    parser.add_argument("--normalization", choices=("zscore", "none"), default="zscore")
    parser.add_argument("--reference-stride", type=int, default=1)
    parser.add_argument("--query-stride", type=int, default=1)

    parser.add_argument("--period", type=int, help="Known gait period; otherwise estimated by autocorrelation.")
    parser.add_argument("--period-min", type=int)
    parser.add_argument("--period-max", type=int)
    parser.add_argument("--period-burn-in", type=int, default=100)
    parser.add_argument("--template-trajectory", type=int, default=0)
    parser.add_argument("--template-start", type=int, default=100)
    parser.add_argument("--phase-probe-steps", type=int, default=8)
    parser.add_argument("--phase-tolerance", type=float, default=0.10, help="Close-phase radius in cycles.")
    parser.add_argument(
        "--mismatch-offsets",
        type=parse_float_list,
        default=parse_float_list("0,0.125,0.25,0.375,0.5"),
        help="Controlled circular offsets in cycles.",
    )
    parser.add_argument("--mismatch-width", type=float, default=0.04, help="Half-width of each mismatch bin.")

    parser.add_argument("--criterion", choices=("ot", "l2", "cosine"), default="ot")
    parser.add_argument("--ot-cost", choices=("l2", "huber", "cosine"), default="l2")
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--ot-iters", type=int, default=60)
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument(
        "--oracle-steps",
        type=int,
        default=8,
        help="Aligned feature steps used by the independent all-window oracle; 0 uses the full horizon.",
    )
    parser.add_argument("--oracle-batch-size", type=int, default=32)
    parser.add_argument("--criterion-batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "reference_trajectories",
        "query_trajectories",
        "num_query_windows",
        "horizon",
        "reference_stride",
        "query_stride",
        "phase_probe_steps",
        "oracle_batch_size",
        "criterion_batch_size",
        "k",
        "ot_iters",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.tau <= 0:
        raise ValueError("--tau must be positive")
    if args.ot_eps <= 0:
        raise ValueError("--ot-eps must be positive")
    if args.oracle_steps < 0:
        raise ValueError("--oracle-steps cannot be negative")
    if args.template_start < 0:
        raise ValueError("--template-start cannot be negative")
    if args.period is not None and args.period <= 1:
        raise ValueError("--period must be greater than one")
    if args.period_min is not None and args.period_min <= 1:
        raise ValueError("--period-min must be greater than one")
    if args.period_max is not None and args.period_max <= 1:
        raise ValueError("--period-max must be greater than one")
    if not 0 <= args.phase_tolerance <= 0.5:
        raise ValueError("--phase-tolerance must lie in [0, 0.5]")
    if not 0 <= args.mismatch_width <= 0.5:
        raise ValueError("--mismatch-width must lie in [0, 0.5]")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    defaults = ENV_DEFAULTS[args.env]
    expert_path = resolve_path(args.expert_path or Path(defaults["expert_path"]))
    rollout_path = resolve_path(args.rollout_path) if args.rollout_path else None
    input_type = args.input_type or defaults["input_type"]
    period_min = args.period_min or defaults["period_min"]
    period_max = args.period_max or defaults["period_max"]
    device = choose_device(args.device)
    output_dir = resolve_path(args.output_dir or Path("workdir") / "phase_mismatch_diagnostic" / args.env)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    expert_observations, expert_lengths = load_trajectories(expert_path, args.obs_key)
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
                f"Held-out mode needs {query_stop} expert trajectories, but only "
                f"{expert_observations.shape[0]} are available"
            )
        query_observations = expert_observations[query_start:query_stop]
        query_lengths = expert_lengths[query_start:query_stop]
        query_source = "held_out_expert"
    else:
        rollout_observations, rollout_lengths = load_trajectories(rollout_path, args.obs_key)
        count = min(args.query_trajectories, rollout_observations.shape[0])
        query_observations = rollout_observations[:count]
        query_lengths = rollout_lengths[:count]
        query_source = "agent_rollout"

    if reference_observations.shape[-1] != query_observations.shape[-1]:
        raise ValueError(
            f"Reference feature dimension {reference_observations.shape[-1]} does not match "
            f"query dimension {query_observations.shape[-1]}"
        )

    mean, std = valid_statistics(reference_observations, reference_lengths)
    phase_dimensions = std > 1e-4
    if not phase_dimensions.any():
        raise ValueError("All reference observation dimensions are effectively constant")
    phase_reference = (reference_observations[..., phase_dimensions] - mean[phase_dimensions]) / std[
        phase_dimensions
    ].clamp_min(1e-6)

    if args.period is None:
        period, period_candidates = estimate_period(
            phase_reference,
            reference_lengths,
            period_min,
            period_max,
            args.period_burn_in,
        )
    else:
        period = args.period
        period_candidates = []
    if period <= 1:
        raise ValueError("The gait period must be greater than one")

    template_trajectory = args.template_trajectory
    if not 0 <= template_trajectory < reference_observations.shape[0]:
        raise ValueError("--template-trajectory is outside the reference bank")
    template_length = int(reference_lengths[template_trajectory].item())
    template_start = min(args.template_start, template_length - period)
    if template_start < 0:
        raise ValueError("The selected template trajectory is shorter than the gait period")
    cycle = phase_reference[template_trajectory, template_start : template_start + period]
    probe_steps = min(args.phase_probe_steps, period, args.horizon)
    signatures = template_signatures(cycle, probe_steps)

    normalized_reference = normalize_observations(reference_observations, mean, std, args.normalization)
    normalized_query = normalize_observations(query_observations, mean, std, args.normalization)
    raw_window_size = args.horizon + 1 if input_type == "state_state" else args.horizon

    reference_raw_windows, reference_trajectory_ids, reference_starts = extract_windows(
        normalized_reference,
        reference_lengths,
        raw_window_size,
        args.reference_stride,
    )
    reference_phase_windows, phase_trajectory_ids, phase_starts = extract_windows(
        phase_reference,
        reference_lengths,
        raw_window_size,
        args.reference_stride,
    )
    if not torch.equal(reference_trajectory_ids, phase_trajectory_ids) or not torch.equal(reference_starts, phase_starts):
        raise RuntimeError("Internal reference-window metadata mismatch")

    query_raw_windows, query_trajectory_ids, query_starts = extract_windows(
        normalized_query,
        query_lengths,
        raw_window_size,
        args.query_stride,
    )
    requested_queries = min(args.num_query_windows, query_raw_windows.shape[0])
    query_selection = torch.randperm(query_raw_windows.shape[0], generator=generator)[:requested_queries]
    query_raw_windows = query_raw_windows[query_selection]
    query_trajectory_ids = query_trajectory_ids[query_selection]
    query_starts = query_starts[query_selection]

    reference_features = make_features(reference_raw_windows, input_type).to(device)
    query_features = make_features(query_raw_windows, input_type).to(device)
    signatures_device = signatures.to(device)
    reference_phase_windows = reference_phase_windows.to(device)
    reference_phases, phase_assignment_costs = assign_phases(
        reference_phase_windows,
        signatures_device,
        probe_steps,
        args.oracle_batch_size,
    )
    reference_phases = reference_phases.cpu()
    phase_assignment_costs = phase_assignment_costs.cpu()

    oracle_steps = args.horizon if args.oracle_steps == 0 else min(args.oracle_steps, args.horizon)
    oracle_queries = query_features[:, :oracle_steps].reshape(query_features.shape[0], -1)
    oracle_references = reference_features[:, :oracle_steps].reshape(reference_features.shape[0], -1)
    oracle_indices_device, oracle_costs_device = nearest_rows(
        oracle_queries,
        oracle_references,
        args.oracle_batch_size,
    )
    oracle_indices = oracle_indices_device.cpu()
    oracle_costs = oracle_costs_device.cpu()
    oracle_phases = reference_phases[oracle_indices]

    criterion = build_criterion(args)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    random_candidates = torch.randint(
        reference_features.shape[0],
        (query_features.shape[0], args.k),
        generator=generator,
    )
    random_costs = candidate_costs(
        query_features,
        reference_features,
        random_candidates,
        criterion,
        args.criterion_batch_size,
    )
    for pool_name, candidates, costs in (
        ("random_k1", random_candidates[:, :1], random_costs[:, :1]),
        (f"random_k{args.k}", random_candidates, random_costs),
    ):
        summary, details = summarize_pool(
            pool_name,
            None,
            candidates,
            costs,
            oracle_indices,
            oracle_costs,
            oracle_phases,
            reference_phases,
            reference_trajectory_ids,
            reference_starts,
            period,
            args.tau,
            args.phase_tolerance,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)

    for mismatch in args.mismatch_offsets:
        controlled_candidates = sample_controlled_candidates(
            oracle_phases,
            reference_phases,
            period,
            mismatch,
            args.mismatch_width,
            args.k,
            generator,
        )
        controlled_costs = candidate_costs(
            query_features,
            reference_features,
            controlled_candidates,
            criterion,
            args.criterion_batch_size,
        )
        summary, details = summarize_pool(
            f"controlled_{mismatch:.3f}",
            mismatch,
            controlled_candidates,
            controlled_costs,
            oracle_indices,
            oracle_costs,
            oracle_phases,
            reference_phases,
            reference_trajectory_ids,
            reference_starts,
            period,
            args.tau,
            args.phase_tolerance,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)

    metadata = {
        "environment": args.env,
        "expert_path": str(expert_path),
        "rollout_path": str(rollout_path) if rollout_path else None,
        "query_source": query_source,
        "input_type": input_type,
        "criterion": args.criterion,
        "ot_cost": args.ot_cost if args.criterion == "ot" else None,
        "normalization": args.normalization,
        "observation_dim": reference_observations.shape[-1],
        "phase_dimensions": int(phase_dimensions.sum().item()),
        "reference_trajectories": reference_observations.shape[0],
        "query_trajectories": query_observations.shape[0],
        "reference_windows": reference_features.shape[0],
        "query_windows": query_features.shape[0],
        "horizon": args.horizon,
        "oracle_steps": oracle_steps,
        "estimated_period": period,
        "period_local_minima": period_candidates,
        "template_trajectory": template_trajectory,
        "template_start": template_start,
        "phase_probe_steps": probe_steps,
        "phase_assignment_l2_mean": finite_mean(phase_assignment_costs),
        "phase_tolerance": args.phase_tolerance,
        "k": args.k,
        "tau": args.tau,
        "seed": args.seed,
        "device": str(device),
        "query_samples": [
            {
                "query_id": index,
                "trajectory": int(query_trajectory_ids[index].item()),
                "start": int(query_starts[index].item()),
            }
            for index in range(query_features.shape[0])
        ],
        "interpretation": (
            "The oracle is the minimum aligned-L2 window over the complete reference bank under the selected normalization. "
            "The highest-weight softmin segment equals the hard-min segment for a fixed candidate pool; "
            "soft/hard differ in weight mass, effective cost, and training gradients. Cost-derived rewards are "
            "diagnostic proxies, not environment returns."
        ),
    }

    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_query.csv", detail_rows)
    with (output_dir / "results.json").open("w") as file:
        json.dump({"metadata": metadata, "results": summary_rows}, file, indent=2)
    plot_mismatch_curve(output_dir / "phase_mismatch.png", summary_rows)

    print(f"Environment: {args.env}")
    print(f"Query source: {query_source}")
    print(f"Estimated gait period: {period} steps")
    print(f"Reference/query windows: {reference_features.shape[0]}/{query_features.shape[0]}")
    print()
    header = f"{'pool':<20} {'coverage':>10} {'selected':>10} {'phase err':>10} {'entropy':>10} {'hard cost':>11}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['pool']:<20} "
            f"{row['candidate_coverage_fraction']:>10.3f} "
            f"{row['selected_close_fraction']:>10.3f} "
            f"{row['highest_weight_phase_error_mean']:>10.3f} "
            f"{row['soft_normalized_entropy_mean']:>10.3f} "
            f"{row['hard_cost_mean']:>11.4f}"
        )
    print(f"\nWrote diagnostic outputs to {output_dir}")


if __name__ == "__main__":
    main()
