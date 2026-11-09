"""Rebuttal figures: alignment-objective comparison and the transport-plan
gradient ablation.

Answers, with runs rather than prose:

* Q9cS "add DTW-based baselines / justify the necessity of OT" and "would DTW
  handle these near-diagonal plans anyway" -> soft-DTW and subsequence soft-DTW
  substituted for the Sinkhorn coupling in an otherwise identical pipeline
  (same ground cost, per-step cost scale, best-of-K softmin, reward transform,
  critic, backbone).
* Q9cS "best-of-K is a sampled subsequence soft-DTW" -> subseq at K=1 vs K=8.
* LMQH "Fig. 13 shows little difference between K=1 and K=8 for OT-Cos" -> the
  two low-temperature OT rows, which test whether beta was simply off the scale
  of the segment scores.
* LMQH "the code detaches the transport plan" -> gradpath: fixed-coupling vs
  differentiating through the Sinkhorn iterations, both arms run fresh.

Baselines are the FOCUS runs reported in the paper, read from
grouped_exp_urls.yaml so the provenance is the same as every other figure.

    python rebuttal_plots.py           # writes plotter/rebuttal/
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import yaml

from plotter import ENV_NAMES, EXPERTS, MAX_STEPS_PER_ENV, compute_center_and_band
from wandb_api import API

# The paper's figures read train_scores/episode_rewards against _step (which is
# agent steps) and truncate at MAX_STEPS_PER_ENV. The summary's
# eval_scores/episode_rewards cannot be used here: the reported runs were
# launched with max_agent_steps=1e8 and stopped on wall-clock, so they ran to
# ~26M steps on SNU Humanoid while the rebuttal runs stop at exactly 15M.
# Comparing those summaries would compare different interaction budgets.
CURVE_METRIC = "train_scores/episode_rewards"
TAIL = 5  # average the last few logged points so one noisy eval does not decide a seed

# A run that exits on the JobClock's 30-minute safety buffer (SHAC.train breaks
# on "Not enough time left for another step") exits cleanly, so W&B marks it
# "finished" even though it never reached max_agent_steps. Scoring its tail
# would put an ~11M-step policy in the same cell as a 15M-step one. Require the
# history to actually reach the budget before a seed counts.
MIN_COVERAGE = 0.98
SHORT_RUNS = []  # (project, run name, env, last logged step, cap)

PLOTTER_DIR = Path(__file__).resolve().parent
OUT_DIR = PLOTTER_DIR / "rebuttal"
ENTITY = "jacr"
ENVS = ["hopper", "ant", "humanoid", "snu_humanoid"]


# Row label <- (imitation_loss_type, K, tau, cost, detach_plan)
def row_label(ot):
    lt = ot.get("imitation_loss_type")
    k = ot.get("loss_best_of_k_k")
    tau = ot.get("loss_best_of_k_tau")
    cost = ot.get("loss_ot_cost_type")
    det = ot.get("loss_ot_detach_plan")
    if lt == "softdtw":
        return f"FOCUS-softDTW (K={k})"
    if lt == "softdtw_subseq":
        return f"FOCUS-subseqDTW (K={k})"
    if lt == "ot":
        if det is False:
            return "OT-L2, through-plan grad"
        if tau is not None and float(tau) != 0.5:
            # Same Sinkhorn OT as the reported runs; only the best-of-K
            # temperature differs, so the label has to say so or the bar reads
            # as a different matching method.
            return f"FOCUS-OT-{'Cos' if cost=='cosine' else 'L2'} (beta={tau})"
        return "OT-L2, fixed coupling"
    return str(lt)


ALIGNMENT_ORDER = [
    "FOCUS-OT-L2 (reported, beta=0.5)",
    "FOCUS-OT-Cos (reported, beta=0.5)",
    "FOCUS-softDTW (K=8)",
    "FOCUS-subseqDTW (K=8)",
    "FOCUS-subseqDTW (K=1)",
    "FOCUS-OT-L2 (beta=0.02)",
    "FOCUS-OT-Cos (beta=0.01)",
]
COLORS = {
    # green  = Sinkhorn OT at the reported temperature (the paper's method)
    # blue   = Sinkhorn replaced by an ordered alignment
    # orange = the same Sinkhorn OT, only the best-of-K temperature changed
    "FOCUS-OT-L2 (reported, beta=0.5)": "#479A5F",
    "FOCUS-OT-Cos (reported, beta=0.5)": "#A0C75C",
    "FOCUS-softDTW (K=8)": "#2B6CB0",
    "FOCUS-subseqDTW (K=8)": "#6AA3DD",
    "FOCUS-subseqDTW (K=1)": "#AFCDEA",
    "FOCUS-OT-L2 (beta=0.02)": "#B4651D",
    "FOCUS-OT-Cos (beta=0.01)": "#E0A569",
    "OT-L2, fixed coupling": "#479A5F",
    "OT-L2, through-plan grad": "#C5263D",
}


def run_config(run):
    """A run's config as a plain nested dict.

    The W&B client sometimes hands back `config` as a JSON string rather than a
    dict, and entries can be wrapped as {"value": ..., "desc": ...}. Accessing
    it directly raises `AttributeError: 'str' object has no attribute 'get'`
    partway through a refresh, which loses the whole fetch.
    """
    cfg = run.config
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            return {}
    out = {}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and set(v) <= {"value", "desc"}:
            v = v.get("value")
        out[k] = v
    return out


def final_at_budget(run, env):
    """Final return at the env's interaction budget, matching plotter.py."""
    cap = MAX_STEPS_PER_ENV[env]
    try:
        h = run.history(keys=[CURVE_METRIC], samples=100000, x_axis="_step")
    except Exception:
        return None
    if h is None or h.empty or CURVE_METRIC not in h:
        return None
    h = h.dropna(subset=[CURVE_METRIC])
    sub = h[h["_step"] <= cap]
    if sub.empty:
        return None
    reached = float(sub["_step"].max())
    if reached < MIN_COVERAGE * cap:
        SHORT_RUNS.append((run.project, run.name, env, reached, cap))
        return None
    return float(sub[CURVE_METRIC].tail(TAIL).mean())


def _scores(project, env):
    """{label: {seed: final return at budget}} for finished runs."""
    out = defaultdict(dict)
    try:
        runs = list(API.runs(f"{ENTITY}/{project}", per_page=200))
    except Exception as exc:  # project absent
        print(f"  ! {project}: {exc}")
        return out
    for r in runs:
        if r.state != "finished":
            continue
        v = final_at_budget(r, env)
        if v is None:
            continue
        cfg = run_config(r)
        agent = cfg.get("agent") or {}
        ot = (agent.get("otil") or {}) if isinstance(agent, dict) else {}
        seed = cfg.get("seed")
        label = row_label(ot)
        ts = r.summary.get("_timestamp", 0)
        prev = out[label].get(seed)
        if prev is None or ts > prev[1]:
            out[label][seed] = (v, ts)
    return {lab: {s: v[0] for s, v in d.items()} for lab, d in out.items()}


def fetch_reported_baselines():
    """FOCUS-OT-L2 / OT-Cos from the groups used for the paper's figures."""
    groups = yaml.safe_load(open(PLOTTER_DIR / "grouped_exp_urls.yaml"))
    want = {"focus-ot-l2": "FOCUS-OT-L2 (reported, beta=0.5)",
            "focus-ot-cos": "FOCUS-OT-Cos (reported, beta=0.5)"}
    data = defaultdict(dict)
    for env, methods in groups.items():
        for method, url in methods.items():
            key = method.lower().replace("_", "-")
            if key not in want:
                continue
            parts = url.split("/")
            ent, proj, grp = parts[3], parts[4], parts[6]
            runs = API.runs(f"{ent}/{proj}", filters={"group": grp})
            vals = {}
            for r in runs:
                if r.state != "finished":
                    continue
                v = final_at_budget(r, env)
                if v is not None:
                    vals[run_config(r).get("seed")] = v
            if vals:
                data[env][want[key]] = vals
    return data


def collect():
    data = defaultdict(dict)  # env -> label -> [scores]
    print("Reported FOCUS baselines:")
    for env, d in fetch_reported_baselines().items():
        for lab, vals in d.items():
            data[env][lab] = vals
            print(f"  {env:14s} {lab:28s} n={len(vals)}")
    print("Soft-DTW / temperature rows:")
    for env in ENVS:
        got = _scores(f"softdtw_OTIL_SAPO-dflex_{env}-slurm", env)
        for lab, d in got.items():
            data[env][lab] = d
            print(f"  {env:14s} {lab:28s} n={len(d)}")
    print("Gradient-path ablation:")
    grad = defaultdict(dict)
    for env in ENVS:
        got = _scores(f"gradpath_OTIL_SAPO-dflex_{env}-slurm", env)
        for lab, d in got.items():
            grad[env][lab] = d
            print(f"  {env:14s} {lab:28s} n={len(d)}")
    return data, grad


def norm_stats(vals, env, band="95ci"):
    """vals is {seed: score}; returns expert-normalized mean and CI."""
    e = EXPERTS[env]["mean"]
    arr = np.asarray(sorted(vals.values()) if isinstance(vals, dict) else vals, dtype=float) / e
    c, lo, hi = compute_center_and_band(arr, center_stat="mean", band=band)
    return float(c), float(lo), float(hi), len(arr)


def write_table(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.with_suffix('.csv')}")


def plot_alignment(data):
    labels = [l for l in ALIGNMENT_ORDER if any(l in data[e] for e in ENVS)]
    fig, axes = plt.subplots(1, len(ENVS), figsize=(4.0 * len(ENVS), 4.2), sharey=True)
    rows = []
    for ax, env in zip(axes, ENVS):
        present = [l for l in labels if l in data[env]]
        xs = np.arange(len(present))
        for i, lab in enumerate(present):
            c, lo, hi, n = norm_stats(data[env][lab], env)
            rows.append(dict(environment=env, method=lab, mean=c, ci_low=lo, ci_high=hi, n_seeds=n))
            ax.bar(i, c, color=COLORS.get(lab, "#888888"), width=0.72,
                   edgecolor="white", linewidth=0.8)
            ax.errorbar(i, c, yerr=[[max(c - lo, 0)], [max(hi - c, 0)]], fmt="none",
                        ecolor="#333333", elinewidth=1.1, capsize=3)
        ax.axhline(1.0, color="#616161", ls="--", lw=1.0, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(present, rotation=38, ha="right", fontsize=7.5)
        ax.set_title(ENV_NAMES[env], fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", color="#E3E3E3", lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("expert-normalized return")
    fig.suptitle(
        "Every bar is FOCUS with one component changed   "
        "(6 seeds, mean with 95% bootstrap CI)\n"
        "green: Sinkhorn OT as reported   |   "
        "blue: Sinkhorn replaced by an ordered alignment   |   "
        "orange: same Sinkhorn OT, best-of-K temperature changed",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "alignment_objective.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"wrote {p}")
    write_table(rows, OUT_DIR / "alignment_objective")
    return rows


def plot_gradpath(grad):
    envs = [e for e in ENVS if grad.get(e)]
    if not envs:
        print("  (no gradpath runs)")
        return []
    fig, axes = plt.subplots(1, len(envs), figsize=(max(7.0, 4.0 * len(envs)), 4.2), squeeze=False)
    rows = []
    order = ["OT-L2, fixed coupling", "OT-L2, through-plan grad"]
    for ax, env in zip(axes[0], envs):
        present = [l for l in order if l in grad[env]]
        for i, lab in enumerate(present):
            c, lo, hi, n = norm_stats(grad[env][lab], env)
            rows.append(dict(environment=env, method=lab, mean=c, ci_low=lo, ci_high=hi, n_seeds=n))
            ax.bar(i, c, color=COLORS[lab], width=0.6, edgecolor="white")
            ax.errorbar(i, c, yerr=[[max(c - lo, 0)], [max(hi - c, 0)]], fmt="none",
                        ecolor="#333333", elinewidth=1.1, capsize=4)
            ax.text(i, c + 0.03, f"{c:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(["fixed coupling\n(as implemented)",
                            "through Sinkhorn\n(as described in paper)"][:len(present)],
                           fontsize=8.5)
        ax.axhline(1.0, color="#616161", ls="--", lw=1.0, zorder=0)
        ax.set_ylim(0, 1.15)
        ax.set_title(ENV_NAMES[env], fontsize=11)
        ax.grid(axis="y", color="#E3E3E3", lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("expert-normalized return")
    fig.suptitle("Transport-plan gradient\nboth arms run fresh, 6 seeds, mean with 95% bootstrap CI",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = OUT_DIR / "gradient_path.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"wrote {p}")
    write_table(rows, OUT_DIR / "gradient_path")
    return rows


CACHE = OUT_DIR / "_wandb_cache.json"


def collect_cached(force=False):
    """Fetching ~170 run histories takes minutes; cache so figures iterate fast.
    Delete plotter/rebuttal/_wandb_cache.json or pass --refresh to refetch."""
    if CACHE.exists() and not force:
        raw = json.loads(CACHE.read_text())
        print(f"(using cached W&B data from {CACHE}; --refresh to refetch)")
        to_d = lambda d: {e: {m: {int(k): v for k, v in sd.items()} for m, sd in md.items()}
                          for e, md in d.items()}
        SHORT_RUNS[:] = [tuple(x) for x in raw.get("short_runs", [])]
        return to_d(raw["data"]), to_d(raw["grad"])
    data, grad = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(
        {"data": data, "grad": grad, "short_runs": SHORT_RUNS}, default=str))
    return data, grad


if __name__ == "__main__":
    import sys
    data, grad = collect_cached(force="--refresh" in sys.argv)
    if SHORT_RUNS:
        print(f"\n=== dropped {len(SHORT_RUNS)} run(s): stopped short of the budget ===")
        for proj, name, env, reached, cap in sorted(SHORT_RUNS, key=lambda r: (r[2], r[0], r[3])):
            print(f"  {env:14s} {reached/1e6:5.2f}M / {cap/1e6:.0f}M  {proj:44s} {name}")
    else:
        print("\nevery contributing run reached its interaction budget")
    print()
    a_rows = plot_alignment(data)
    g_rows = plot_gradpath(grad)
    print("\n=== expert-normalized means (final return at the env's budget) ===")
    for r in a_rows + g_rows:
        print(f"  {r['environment']:14s} {r['method']:28s} "
              f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  n={r['n_seeds']}")

    # Per-seed dump: aggregate means hide bimodal outcomes, and FOCUS on SNU
    # Humanoid does show seeds that collapse.
    rows = []
    for src, label_set in (("alignment", data), ("gradpath", grad)):
        for env in ENVS:
            for lab, d in sorted(label_set.get(env, {}).items()):
                e = EXPERTS[env]["mean"]
                for seed, v in sorted(d.items()):
                    rows.append(dict(source=src, environment=env, method=lab,
                                     seed=seed, raw=v, normalized=v / e))
    if rows:
        write_table(rows, OUT_DIR / "per_seed")
        print("\n=== seeds below 0.5x expert (collapses) ===")
        bad = [r for r in rows if r["normalized"] < 0.5]
        if not bad:
            print("  none")
        for r in sorted(bad, key=lambda r: (r["environment"], r["method"])):
            print(f"  {r['environment']:14s} {r['method']:28s} seed={r['seed']:<5} "
                  f"{r['normalized']:.3f}")
