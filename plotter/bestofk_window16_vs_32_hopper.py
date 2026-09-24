"""Hopper: does the real RL training K-vs-return curve show a bigger effect
at horizon_len=16 (sub-cycle window, ~43% of Hopper's ~37-step gait period)
than at the original horizon_len=32 (~87% of the period)?

The offline matching-quality diagnostic (scripts/window_k_diagnostic.py)
predicted this: OT's transport plan loses phase-discriminative power once
the window covers close to a full gait cycle, so a shorter, sub-cycle
window should let K keep paying off further. This is the real-training
test of that prediction -- final return, not a matching-quality proxy.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wandb_api import API

from bestofk_ablation import K_VALUES, LABEL, METHOD_ORDER, method_name
from plotter import EXPERTS, compute_center_and_band

ENTITY = "jacr"
PROJECTS = {32: "bestofk_OTIL_SAPO-dflex_hopper-slurm-new", 16: "bestofk_window16_OTIL_SAPO-dflex_hopper-slurm-new"}
FOLDER_TO_SAVE_PLOTS = Path(__file__).resolve().parent / "bestofk_ablation"


def fetch(horizon):
    """method -> list of final returns, one per seed (dedup by most recent, EnsembleCritic only)."""
    project = PROJECTS[horizon]
    runs = list(API.runs(f"{ENTITY}/{project}", per_page=300))
    best_by_seed = {}
    for run in runs:
        if run.state != "finished":
            continue
        cfg = dict(run.config)
        otil = cfg.get("agent", {}).get("otil", {})
        k = otil.get("loss_best_of_k_k")
        cost_type = otil.get("loss_ot_cost_type")
        seed = cfg.get("seed")
        if k is None or cost_type is None or seed is None:
            continue
        method = method_name(int(k), cost_type)
        if method not in METHOD_ORDER:
            continue
        if cfg.get("agent", {}).get("network", {}).get("critic") != "EnsembleCritic":
            continue
        if cfg.get("agent", {}).get("shac", {}).get("horizon_len") != horizon:
            continue
        final_return = run.summary.get("eval_scores/episode_rewards")
        if final_return is None:
            continue
        try:
            final_return = float(final_return)
        except (TypeError, ValueError):
            continue
        if final_return != final_return:  # NaN
            continue
        timestamp = run.summary.get("_timestamp", 0)
        key = (method, seed)
        if key not in best_by_seed or timestamp > best_by_seed[key][0]:
            best_by_seed[key] = (timestamp, final_return)

    by_method = {method: [] for method in METHOD_ORDER}
    for (method, seed), (_, final_return) in best_by_seed.items():
        by_method[method].append(final_return)
    return by_method


def main():
    expert_mean = EXPERTS["hopper"]["mean"]
    data = {horizon: fetch(horizon) for horizon in (16, 32)}
    for horizon in (16, 32):
        print(f"horizon={horizon}:")
        for method in METHOD_ORDER:
            print(f"  {method}: {len(data[horizon][method])} seed(s)")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # Color by horizon (the variable under test), matching the same horizon
    # -> color mapping used in docs/phase_rescue/window_k_diagnostic.png so
    # the two figures read consistently side by side. Cost is encoded by
    # linestyle/marker instead.
    color = {16: "#74C476", 32: "#31A354"}
    linestyle = {"l2": "-", "cosine": "--"}
    marker = {"l2": "o", "cosine": "s"}

    for horizon in (16, 32):
        for cost in ("l2", "cosine"):
            ks, centers, lowers, uppers = [], [], [], []
            for k in K_VALUES:
                values = data[horizon].get(method_name(k, cost), [])
                if not values:
                    continue
                values = np.asarray(values, dtype=np.float64) / expert_mean
                center, lower, upper = compute_center_and_band(values[:, None], center_stat="mean", band="95ci")
                ks.append(k)
                centers.append(center[0])
                lowers.append(lower[0])
                uppers.append(upper[0])
            if not ks:
                continue
            centers, lowers, uppers = np.array(centers), np.array(lowers), np.array(uppers)
            yerr = np.vstack([centers - lowers, uppers - centers])
            ax.errorbar(
                ks, centers, yerr=yerr, marker=marker[cost], markersize=7, linewidth=2, capsize=3,
                linestyle=linestyle[cost], color=color[horizon],
                label=f"horizon={horizon}, {LABEL[cost]}",
            )

    ax.axhline(1.0, color="#616161", linestyle=":", linewidth=1, label="Expert")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Best-of-K pool size (K)")
    ax.set_ylabel("Mean final return / expert return")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()

    out_path = FOLDER_TO_SAVE_PLOTS / "bestofk_window16_vs_32_hopper.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
