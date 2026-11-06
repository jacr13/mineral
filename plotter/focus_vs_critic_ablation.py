"""Main-table FOCUS-OT-L2 / FOCUS-OT-Cos runs vs. the reward-ablation runs
critic-exp-l2 / critic-logexp-cos, on identical processing (25-point smoothing,
per-seed last-value endpoint, bootstrap 95% CI) so the two run sets are directly comparable.

Ablation runs: critic_no_critic.py's wandb projects (new_v2_critic_OTIL_SAPO-*).
Main runs: the enabled entries of grouped_exp_logdirs.yaml (the runs behind the main table).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import critic_no_critic as cnc
from plotter import ENV_NAMES, EXPERTS

SMOOTH_WINDOW = 25
ENVS = ["hopper", "ant", "humanoid", "snu_humanoid"]
OUT_DIR = cnc.FOLDER_TO_SAVE_PLOTS

# (label, source, key): ablation method name or grouped_exp_logdirs algorithm name
SERIES = [
    ("FOCUS-OT-L2 (main runs)", "main", "FOCUS-ot-l2", "#479A5F", "-"),
    ("critic-exp-l2 (ablation runs)", "ablation", "critic-exp-l2", "#2a78d6", "--"),
    ("FOCUS-OT-Cos (main runs)", "main", "FOCUS-ot-cos", "#A0C75C", "-"),
    ("critic-logexp-cos (ablation runs)", "ablation", "critic-logexp-cos", "#eda100", "--"),
]
MAIN_ALIASES = {"FOCUS-ot-l2": ("FOCUS-ot-l2", "FOCUS-OT-l2"), "FOCUS-ot-cos": ("FOCUS-ot-cos", "FOCUS-OT-cos")}


def load_main_runs(env, algo):
    groups = yaml.safe_load(open(Path(__file__).parent / "grouped_exp_logdirs.yaml"))[env]
    for name in MAIN_ALIASES[algo]:
        if name in groups:
            return groups[name]
    return []


def build_series(env):
    """label -> list of per-seed curves on one shared grid (NaN past a seed's end)."""
    raw = {label: [] for label, *_ in SERIES}

    for label, source, key, *_ in SERIES:
        if source == "main":
            for run in load_main_runs(env, key):
                ep_rew, steps, times = cnc.fetch_run_history(run["project"], run["run_id"])
                clipped = cnc.clip_to_env_cap(ep_rew, steps, times, env, x_axis="steps")
                if clipped is not None:
                    raw[label].append(clipped)

    project = cnc.PROJECTS[env]
    for run in cnc.API.runs(f"{cnc.ENTITY}/{project}", per_page=200):
        if run.state != "finished" or run.id in cnc.EXCLUDED_RUN_IDS:
            continue
        otil = dict(run.config).get("agent", {}).get("otil", {})
        if otil.get("critic_reward_shapping") is not True:
            continue
        method = cnc.method_name(otil.get("critic_disabled"), otil.get("critic_reward_mapping"), otil.get("loss_ot_cost_type"))
        for label, source, key, *_ in SERIES:
            if source == "ablation" and method == key:
                ep_rew, steps, times = cnc.fetch_run_history(project, run.id)
                clipped = cnc.clip_to_env_cap(ep_rew, steps, times, env, x_axis="steps")
                if clipped is not None:
                    raw[label].append(clipped)

    # One shared grid per environment, so every series is on the same x-axis.
    all_runs = [r for runs in raw.values() for r in runs]
    shared_max = max(x[-1] for _, x in all_runs)
    grid = np.linspace(0, shared_max, num=1000)
    out = {}
    for label, runs in raw.items():
        curves = []
        for ep_rew, x in runs:
            n_valid = max(int(np.searchsorted(grid, x[-1], side="right")), 1)
            curve = cnc.moving_average(np.interp(grid[:n_valid], x, ep_rew), SMOOTH_WINDOW)
            curves.append(np.concatenate([curve, np.full(grid.size - n_valid, np.nan)]))
        out[label] = np.vstack(curves) if curves else None
    return grid, out


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    summary = []
    for ax, env in zip(axes.ravel(), ENVS):
        grid, series = build_series(env)
        expert = EXPERTS[env]["mean"]
        ax.axhline(1.0, color="#616161", linestyle=":", linewidth=1)
        for label, _, _, color, ls in SERIES:
            values = series[label]
            if values is None:
                warnings.warn(f"{env}: no runs for {label}")
                continue
            values = values / expert
            center, lower, upper = cnc.compute_center_and_band_dropout(values, center_stat="mean", band="95ci")
            n = values.shape[0]
            ax.plot(grid, center, color=color, linestyle=ls, linewidth=2, label=f"{label}, n={n}")
            ax.fill_between(grid, lower, upper, color=color, alpha=0.12, linewidth=0)
            last = np.array([v[np.isfinite(v)][-1] for v in values])
            summary.append((env, label, n, last.mean(), np.sort(last).round(3).tolist()))
        ax.set_title(ENV_NAMES[env])
        ax.set_xlabel("Steps")
        ax.margins(x=0)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.supylabel("Normalized Return")
    out = OUT_DIR / "focus_main_vs_critic_ablation_mean95ci_steps_smooth25.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}\n")
    print("Per-seed last smoothed value / expert (mean, sorted seeds):")
    for env, label, n, mean, seeds in summary:
        print(f"  {env:13s} {label:36s} n={n} mean={mean:.3f} seeds={seeds}")


if __name__ == "__main__":
    main()
