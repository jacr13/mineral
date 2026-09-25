"""Two extra views on the best-of-K ablation, since final RL return alone
(bestofk_ablation.py) doesn't show a clean K effect -- it's a noisy,
downstream signal with a lot of RL-training variance on top of whatever K
actually changes.

1. Best-of-K matching cost vs K: uses train_stats/Lk_min (the true min cost
   across the K sampled candidates), NOT train_stats/imitation_loss.
   imitation_loss is a softmin over K (mineral/agents/otil/best_of_k.py:
   loss = -tau*logsumexp(-Lk/tau, dim=1)), which mechanically shifts down by
   ~-tau*log(K) as K grows regardless of any real match-quality improvement
   (e.g. tau=0.5, K=1->16 gives -0.5*log(16)=-1.39 -- matches almost exactly
   what a first pass of this plot showed before this was caught). Lk_min
   doesn't have that bias, so a real K effect here is genuine order-statistics
   improvement (min of more sampled candidates), not a logging artifact.
   L2 and cosine costs live on different scales, so each variant's series is
   shown relative to its own K=1 value (ratio), comparable across variants
   and environments despite the different underlying units.
2. Final return pooled across all 4 envs: each (K, cost) point in
   bestofk_ablation.py's per-env plot only has 6 seeds; pooling
   expert-normalized returns across Hopper/Ant/Humanoid/SNU quadruples that
   to 24, which should tighten the CI enough to reveal a real effect if one
   exists at that precision.

Reuses bestofk_ablation.py's project list, method naming, filtering (correct
EnsembleCritic/SAPO config only, dedup reruns by keeping the most recent),
and color scheme so this stays visually/logically consistent with it.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wandb_api import API

from bestofk_ablation import COLOR, ENVS, K_VALUES, LABEL, METHOD_ORDER, PROJECTS, method_name
from plotter import ENV_NAMES, EXPERTS, compute_center_and_band

FOLDER_TO_SAVE_PLOTS = Path(__file__).resolve().parent / "bestofk_ablation"


def _to_finite_float(value):
    """None/missing -> None. Non-finite (incl. wandb's JSON-safe "NaN" string) -> None. Else float(value)."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fetch_extended_data(envs=None):
    """env -> method -> list of {"final_return", "lk_min"} per seed.

    Same filtering/dedup rules as bestofk_ablation.fetch_data: only
    finished, EnsembleCritic (correct SAPO config) runs, one entry per
    (method, seed) keeping the most recent by _timestamp if a seed was
    resubmitted.
    """
    envs = envs or ENVS
    data = {}
    for env_name in envs:
        project = PROJECTS[env_name]
        print(f"Fetching run list for {env_name}: jacr/{project}")
        runs = list(API.runs(f"jacr/{project}", per_page=200))

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

            final_return = _to_finite_float(run.summary.get("eval_scores/episode_rewards"))
            lk_min = _to_finite_float(run.summary.get("train_stats/Lk_min"))
            # A handful of runs logged Lk_min as NaN -- sometimes a float NaN,
            # sometimes wandb's JSON-safe string "NaN" (e.g. humanoid/
            # snu_humanoid k=1 cosine, seed 100/1000/1200) -- not None, so it
            # would silently NaN-poison the whole variant's K=1 baseline (and
            # every ratio derived from it) unless filtered. Filtered per
            # metric, not per run: those same runs have a perfectly valid
            # final_return, which the pooled-return plot should still use.
            if final_return is None and lk_min is None:
                continue

            timestamp = run.summary.get("_timestamp", 0)
            key = (method, seed)
            if key not in best_by_seed or timestamp > best_by_seed[key][0]:
                best_by_seed[key] = (timestamp, final_return, lk_min)

        by_method = {method: [] for method in METHOD_ORDER}
        for (method, seed), (_, final_return, lk_min) in best_by_seed.items():
            by_method[method].append({"seed": seed, "final_return": final_return, "lk_min": lk_min})

        data[env_name] = by_method
        for method in METHOD_ORDER:
            print(f"  {method}: {len(by_method[method])} seed(s)")

    return data


def plot_lk_min_vs_k(data, output_path):
    """Best-of-K min matching cost (Lk_min) relative to each variant's own K=1 value, per env."""
    env_order = [env for env in ENVS if env in data]
    fig, axes = plt.subplots(1, len(env_order), figsize=(5 * len(env_order), 4), sharex=True)
    if len(env_order) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, env_order):
        env_data = data[env_name]
        for cost in ("l2", "cosine"):
            ks, centers, lowers, uppers = [], [], [], []
            k1_values = [r["lk_min"] for r in env_data.get(method_name(1, cost), []) if r["lk_min"] is not None]
            if not k1_values:
                continue
            k1_mean = float(np.mean(k1_values))
            for k in K_VALUES:
                values = [r["lk_min"] for r in env_data.get(method_name(k, cost), []) if r["lk_min"] is not None]
                if not values:
                    continue
                values = np.asarray(values, dtype=np.float64) / k1_mean
                center, lower, upper = compute_center_and_band(values[:, None], center_stat="mean", band="std")
                ks.append(k)
                centers.append(center[0])
                lowers.append(lower[0])
                uppers.append(upper[0])
            if not ks:
                continue
            centers, lowers, uppers = np.array(centers), np.array(lowers), np.array(uppers)
            yerr = np.vstack([centers - lowers, uppers - centers])
            ax.errorbar(
                ks, centers, yerr=yerr, marker="o", markersize=6, linewidth=2, capsize=3,
                label=LABEL[cost], color=COLOR[cost],
            )

        ax.axhline(1.0, color="#999999", linestyle=":", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Best-of-K pool size (K)")
        ax.set_title(ENV_NAMES.get(env_name, env_name))
        ax.spines[["right", "top"]].set_visible(False)

    axes[0].set_ylabel("Lk_min relative to K=1 (lower = better match)")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("OTIL/FOCUS-SAPO: best-of-K min matching cost (Lk_min) vs K (mean +/- 1 SD across 6 seeds, indexed to K=1)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_pooled_return_vs_k(data, output_path):
    """Expert-normalized final return, pooled across all 4 envs (n=24/point)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for cost in ("l2", "cosine"):
        ks, centers, lowers, uppers, ns = [], [], [], [], []
        for k in K_VALUES:
            pooled = []
            for env_name in ENVS:
                expert_mean = EXPERTS.get(env_name, {}).get("mean")
                if not expert_mean:
                    continue
                records = data.get(env_name, {}).get(method_name(k, cost), [])
                pooled.extend(r["final_return"] / expert_mean for r in records if r["final_return"] is not None)
            if not pooled:
                continue
            values = np.asarray(pooled, dtype=np.float64)
            center, lower, upper = compute_center_and_band(values[:, None], center_stat="mean", band="95ci")
            ks.append(k)
            centers.append(center[0])
            lowers.append(lower[0])
            uppers.append(upper[0])
            ns.append(len(pooled))
        if not ks:
            continue
        centers, lowers, uppers = np.array(centers), np.array(lowers), np.array(uppers)
        yerr = np.vstack([centers - lowers, uppers - centers])
        ax.errorbar(
            ks, centers, yerr=yerr, markersize=7, linewidth=2.2, capsize=4,
            marker={"l2": "o", "cosine": "s"}[cost], linestyle={"l2": "-", "cosine": "--"}[cost],
            label=LABEL[cost], color={"l2": "#2a78d6", "cosine": "#eb6834"}[cost],
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Best-of-K pool size (K)")
    ax.set_ylabel("Mean final return / expert return")
    # Zoomed on purpose (data + CIs span ~0.68-0.96) so the trend is legible;
    # say so in the caption.
    ax.set_ylim(0.65, 1.0)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    data = fetch_extended_data()
    plot_lk_min_vs_k(data, FOLDER_TO_SAVE_PLOTS / "bestofk_lk_min_vs_k.png")
    plot_pooled_return_vs_k(data, FOLDER_TO_SAVE_PLOTS / "bestofk_pooled_return_vs_k.png")
