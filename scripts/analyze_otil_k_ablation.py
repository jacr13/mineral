#!/usr/bin/env python3
"""Summarize an otil_k_ablation.sh sweep: final return vs. best-of-K pool size.

Reads local TensorBoard logs (same tag/layout convention as plotter/plotter.py:
`<logdir>/tb/events.out.tfevents*`, scalar `train_scores/episode_rewards`), so
it works straight off the cluster's synced workdir without a wandb round-trip.

Usage:
    python scripts/analyze_otil_k_ablation.py workdir/OTIL_K_ABLATION_SAC_dflex_ant_20260922_120000
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "train_scores/episode_rewards"
CONDITION_RE = re.compile(r"^(?P<variant>.+)_k(?P<k>\d+)$")

VARIANT_LABELS = {"ot_l2": "FOCUS-OT-L2", "ot_cos": "FOCUS-OT-Cos"}
# Reuse the same hex colors plotter.py already uses for these two variants,
# so this plot stays visually consistent with the rest of the project's figures.
VARIANT_COLORS = {"FOCUS-OT-L2": "#479A5F", "FOCUS-OT-Cos": "#A0C75C"}


def load_run(tb_dir, final_frac):
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if TAG not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(TAG)
    if not events:
        return None
    steps = np.array([e.step for e in events], dtype=np.int64)
    rewards = np.array([e.value for e in events], dtype=np.float64)
    n_tail = max(1, int(len(rewards) * final_frac))
    return {
        "final_return": float(rewards[-n_tail:].mean()),
        "max_step": int(steps[-1]),
        "num_points": int(len(rewards)),
    }


def collect(root, final_frac):
    rows = []
    for condition_dir in sorted(root.iterdir()):
        if not condition_dir.is_dir():
            continue
        match = CONDITION_RE.match(condition_dir.name)
        if not match:
            continue
        variant = VARIANT_LABELS.get(match.group("variant"), match.group("variant"))
        k = int(match.group("k"))
        for seed_dir in sorted(condition_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            tb_dir = seed_dir / "tb"
            if not tb_dir.exists():
                print(f"  skip (no tb/): {seed_dir}")
                continue
            result = load_run(tb_dir, final_frac)
            if result is None:
                print(f"  skip (no '{TAG}' scalar yet): {seed_dir}")
                continue
            seed = seed_dir.name.removeprefix("seed_")
            rows.append({"variant": variant, "k": k, "seed": seed, **result})
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    by_condition = {}
    for row in rows:
        by_condition.setdefault((row["variant"], row["k"]), []).append(row["final_return"])
    summary = []
    for (variant, k), values in sorted(by_condition.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        values = np.array(values)
        summary.append(
            {
                "variant": variant,
                "k": k,
                "n_seeds": len(values),
                "mean_final_return": float(values.mean()),
                "std_final_return": float(values.std()),
            }
        )
    return summary


def plot(summary, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    variants = sorted({row["variant"] for row in summary})
    for variant in variants:
        pts = sorted((row["k"], row["mean_final_return"], row["std_final_return"]) for row in summary if row["variant"] == variant)
        ks = [p[0] for p in pts]
        means = [p[1] for p in pts]
        stds = [p[2] for p in pts]
        color = VARIANT_COLORS.get(variant, None)
        ax.errorbar(ks, means, yerr=stds, marker="o", markersize=6, linewidth=2, capsize=3, label=variant, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({row["k"] for row in summary}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Best-of-K pool size (K)")
    ax.set_ylabel("Final episode return (mean of last window, +/- std across seeds)")
    ax.set_title("OTIL: effect of K on training return")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Ablation root dir, e.g. workdir/OTIL_K_ABLATION_SAC_dflex_ant_<stamp>")
    parser.add_argument("--final-frac", type=float, default=0.05, help="Fraction of the tail of logged points averaged for the final return (default 0.05)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the CSVs/plot (default: <root>/analysis)")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.root} ...")
    rows = collect(args.root, args.final_frac)
    if not rows:
        print("No runs with logged scalars found yet. Nothing to analyze.")
        return

    write_csv(out_dir / "per_seed.csv", rows)
    summary = summarize(rows)
    write_csv(out_dir / "summary.csv", summary)

    print("\nSummary (mean final return +/- std across seeds):")
    for row in summary:
        print(f"  {row['variant']:<14} K={row['k']:<3} n={row['n_seeds']}  {row['mean_final_return']:.1f} +/- {row['std_final_return']:.1f}")

    plot(summary, out_dir / "k_ablation.png")


if __name__ == "__main__":
    main()
