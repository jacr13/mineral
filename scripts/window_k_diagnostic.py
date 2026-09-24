#!/usr/bin/env python3
"""Cheap, CPU-only check of the "smaller window -> K matters more" hypothesis
for Hopper, using the existing paired best-of-K phase-rescue diagnostic
(paired_phase_rescue_diagnostic.py) instead of real training.

Runs the diagnostic at horizon in {8, 16, 32, 64} steps -- 22%, 43%, 87%, and
173% of Hopper's ~37-step gait period respectively (32 is the original, used
by the real bestofk sweep and by scripts/phase_rescue_k_sweep.py) -- for
K in {2,4,8,16} across three FOCUS variants: FOCUS-OT-L2 and FOCUS-OT-Cos
(imitation_loss_type=ot, an OT/Sinkhorn transport plan -- soft-matches
timesteps with no order constraint), and FOCUS-L2 (imitation_loss_type=l2,
SequenceRegressionCriterion -- strict, timestep-aligned MSE, no reordering
at all).

This variant is the control for the mechanism proposed to explain why OT
degrades once horizon >= period: OT's transport plan can align a window with
any phase-shifted copy of the same loop near-for-free once the window
contains a full cycle (same poses, different entry point), eroding phase
discrimination. FOCUS-L2's strict alignment has no such flexibility, so it
should NOT show the same "long window hurts" degradation if that mechanism
is actually what's happening -- if it does degrade the same way, the
mechanism is wrong and something more general is going on.

This measures matching-quality benefit only (candidate coverage/rescue/phase
error on held-out expert windows), not downstream RL return -- same caveat
as phase_rescue_k_sweep.py.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("workdir/window_k_diagnostic/hopper")
REPORT_DIR = Path("docs/phase_rescue")

HORIZONS = [8, 16, 32, 64]
# name -> extra CLI args for paired_phase_rescue_diagnostic.py
VARIANTS = {
    "ot_l2": ["--criterion", "ot", "--ot-cost", "l2"],
    "ot_cos": ["--criterion", "ot", "--ot-cost", "cosine"],
    "l2_no_ot": ["--criterion", "l2"],
}
VARIANT_LABEL = {"ot_l2": "FOCUS-OT-L2", "ot_cos": "FOCUS-OT-Cos", "l2_no_ot": "FOCUS-L2 (no OT)"}
K_VALUES = [2, 4, 8, 16]
# Hardest controlled mismatch in the diagnostic's default sweep (half a
# cycle) -- the condition where Best-of-K has the most room to help.
TARGET_MISMATCH = 0.5


def run_one(horizon, variant, k):
    out_dir = OUTPUT_ROOT / f"h{horizon}_{variant}_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "paired_phase_rescue_diagnostic.py"),
        "--env", "hopper",
        "--horizon", str(horizon),
        "-k", str(k),
        *VARIANTS[variant],
        "--device", "cpu",
        "--seed", "0",
        "--output-dir", str(out_dir),
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=SCRIPTS_DIR.parent, env=env)
    result = json.loads((out_dir / "paired_results.json").read_text())
    row = next(r for r in result["results"] if abs(r["target_mismatch"] - TARGET_MISMATCH) < 1e-6)
    return row


def main():
    rows = []
    for horizon in HORIZONS:
        for variant in VARIANTS:
            for k in K_VALUES:
                print(f"Running horizon={horizon} variant={variant} k={k} ...", flush=True)
                row = run_one(horizon, variant, k)
                rows.append({"horizon": horizon, "variant": variant, "k": k, **row})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "window_k_diagnostic_results.json").write_text(json.dumps(rows, indent=2) + "\n")

    metrics = [
        ("best_of_k_candidate_coverage_fraction", "Candidate coverage (fraction within phase tolerance)"),
        ("rescue_fraction_when_k1_wrong", "Rescue rate (K=1 wrong -> Best-of-K right)"),
        ("best_of_k_phase_error_mean", "Mean phase error (cycles, lower = better)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 4.5))
    # Sequential ramp (light -> dark) since horizon is an ordered magnitude,
    # not a category -- ColorBrewer 4-class Greens.
    color = {8: "#C7E9C0", 16: "#74C476", 32: "#31A354", 64: "#006D2C"}
    linestyle = {"ot_l2": "-", "ot_cos": "--", "l2_no_ot": ":"}

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for horizon in HORIZONS:
            for variant in VARIANTS:
                ks = [row["k"] for row in rows if row["horizon"] == horizon and row["variant"] == variant]
                values = [row[metric_key] for row in rows if row["horizon"] == horizon and row["variant"] == variant]
                if any(v is None for v in values):
                    continue
                ax.plot(
                    ks, values, marker="o", linewidth=2, markersize=6,
                    color=color[horizon], linestyle=linestyle[variant],
                    label=f"horizon={horizon}, {VARIANT_LABEL[variant]}",
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Best-of-K pool size (K)")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label.split("(")[0].strip())
        ax.spines[["right", "top"]].set_visible(False)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"Hopper: window size vs. K benefit (offline diagnostic, target_mismatch={TARGET_MISMATCH} cycles)")
    fig.tight_layout()
    out_path = REPORT_DIR / "window_k_diagnostic.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out_path}")
    print(f"Wrote {REPORT_DIR / 'window_k_diagnostic_results.json'}")


if __name__ == "__main__":
    main()
