"""Imitation-fidelity metrics from saved evaluation rollouts.

Reviewer LMQH asked for evidence that the learned behaviour resembles the
demonstrations, not just that task return is high ("I could not find
qualitative expert-learner rollout comparisons or quantitative measures of
behavioural similarity"). Task return cannot answer that: a policy can score
well while moving quite differently.

Every `run=train_eval` run already saved 128 evaluation episodes next to its
checkpoint, under `<logdir>/demos/*.pt`, so all of this is computed offline --
no simulator, no GPU, and no dependence on the checkpoints that were lost to
the disk purge.

Metrics, each reported for the learner and for the expert demonstrations:

* **gait period** -- autocorrelation-style estimate, reusing
  ``scripts/phase_mismatch_diagnostic.estimate_period`` so it matches the
  paper's Appendix E.4.1 diagnostic rather than introducing a second
  definition. Reported as learner period / expert period; 1.0 means the
  cadence matches.
* **held-out alignment distance**, reported in both directions, since a
  one-sided nearest-neighbour distance rewards mode collapse:
  - *precision* -- learner windows to held-out expert windows. Low means the
    learner's motion looks like expert motion.
  - *coverage* -- held-out expert windows to learner windows. Low means the
    learner reproduces the expert's *range* of motion. A policy repeating one
    stereotyped cycle scores well on precision and badly here.
  Both are normalised by the same quantity measured between disjoint expert
  subsets, so 1.0 means "as close as the expert is to itself" and the numbers
  compare across environments with different state scales.

  **soft-DTW is the primary measure; the OT variant is reported only for
  reference.** ``--self-test`` shows why: time-reversing the expert leaves the
  OT distance at 1.01 while soft-DTW rises to 10.7. OT with uniform marginals
  has no ordering constraint, so it cannot distinguish forward motion from
  backward and is unsuitable as a fidelity measure -- even though it is the
  right object for FOCUS's *training* signal, where the alignment is supposed
  to be order-free.
* **per-dimension Wasserstein-1** -- W1 between learner and expert marginal
  state distributions, averaged over observation dimensions after
  standardising each dimension by the expert's std, so dimensions with large
  units do not dominate.

Usage:
    python plotter/fidelity_metrics.py                  # all envs found locally
    python plotter/fidelity_metrics.py --env hopper
    python plotter/fidelity_metrics.py --max-episodes 32
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

PLOTTER_DIR = Path(__file__).resolve().parent
REPO = PLOTTER_DIR.parent
OUT_DIR = PLOTTER_DIR / "rebuttal"
sys.path.insert(0, str(REPO))

from mineral.agents.otil.best_of_k import log_sinkhorn, soft_dtw  # noqa: E402
from scripts.phase_mismatch_diagnostic import estimate_period  # noqa: E402

EXPERT_DEMOS = {
    "hopper": "experts/demos/DFlex_hopper_demos128_return4812_len1000.pt",
    "ant": "experts/demos/DFlex_ant_demos128_return9318_len1000.pt",
    "humanoid": "experts/demos/DFlex_humanoid_demos128_return8116_len984.pt",
    "snu_humanoid": "experts/demos/DFlex_snu_demos128_return6246_len929.pt",
}
PERIOD_RANGE = {"hopper": (8, 120), "ant": (8, 120), "humanoid": (8, 160), "snu_humanoid": (8, 160)}


# ----------------------------------------------------------------- loading ---

def load_eval_rollouts(path, max_episodes=None):
    """`<logdir>/demos/*.pt` -> [N, T, d] padded obs and [N] lengths."""
    d = torch.load(path, map_location="cpu")
    eps = [d[k] for k in sorted(d)] if isinstance(d, dict) and 0 in d else None
    if eps is None:
        raise ValueError(f"{path}: not a per-episode rollout file")
    if max_episodes:
        eps = eps[:max_episodes]
    seqs = []
    for e in eps:
        o = e["obs"]["obs"] if isinstance(e["obs"], dict) else e["obs"]
        seqs.append(o.float())
    lengths = torch.tensor([s.shape[0] for s in seqs])
    T = int(lengths.max())
    out = torch.zeros(len(seqs), T, seqs[0].shape[-1])
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out, lengths


def load_expert(env, max_episodes=None):
    d = torch.load(REPO / EXPERT_DEMOS[env], map_location="cpu")
    obs = d["obs"]["obs"] if isinstance(d["obs"], dict) else d["obs"]
    lengths = d["lengths"] if "lengths" in d else torch.full((obs.shape[0],), obs.shape[1])
    if max_episodes:
        obs, lengths = obs[:max_episodes], lengths[:max_episodes]
    return obs.float(), lengths


# ----------------------------------------------------------------- metrics ---

def estimate_period_fundamental(obs, lengths, lo, hi, burn_in, tol=1.25):
    """Period, preferring the fundamental over its harmonics.

    ``estimate_period`` returns the lag with the lowest autocorrelation-style
    score, but for a periodic gait every multiple of the true period is also a
    local minimum -- on Hopper the expert's minima sit at 37, 74 and 110. The
    global minimum therefore lands on a harmonic often enough to make a raw
    ratio meaningless (observed learner periods of 67, 74 and 103 against an
    expert 37).

    Among the detected local minima we take the *smallest* lag whose score is
    within ``tol`` of the best, which selects the fundamental whenever the
    harmonics are comparably deep.
    """
    best, minima = estimate_period(obs, lengths, lo, hi, burn_in)
    if not minima:
        return best, best, minima
    best_score = min(sc for _, sc in minima)
    ok = [lag for lag, sc in minima if sc <= tol * best_score]
    return (min(ok) if ok else best), best, minima


def _windows(obs, lengths, size, n_max, rng):
    """Random windows of `size` steps, sampled across trajectories."""
    out = []
    for i, L in enumerate(lengths.tolist()):
        if L < size:
            continue
        for _ in range(max(1, n_max // max(1, len(lengths)))):
            s = rng.integers(0, L - size + 1)
            out.append(obs[i, s : s + size])
    if not out:
        return None
    w = torch.stack(out)
    if w.shape[0] > n_max:
        idx = rng.choice(w.shape[0], n_max, replace=False)
        w = w[idx]
    return w


def alignment_distance(A, B, mu, sd, kind="dtw", gamma=0.1, eps=0.1, iters=60, chunk=64,
                       device="cpu"):
    """Mean alignment cost from each window in A to its nearest window in B.

    Both are standardised by the *expert's* per-dimension mean/std so the cost
    is on one scale regardless of the environment's units.
    """
    A = ((A - mu) / sd).to(device)
    B = ((B - mu) / sd).to(device)
    nA, T, _ = A.shape
    best = torch.full((nA,), float("inf"), device=device)
    for j0 in range(0, B.shape[0], chunk):
        Bc = B[j0 : j0 + chunk]
        # pairwise cost matrices between every A window and every B window
        a = A.unsqueeze(1).expand(nA, Bc.shape[0], T, A.shape[-1]).reshape(-1, T, A.shape[-1])
        b = Bc.unsqueeze(0).expand(nA, Bc.shape[0], T, B.shape[-1]).reshape(-1, T, B.shape[-1])
        C = ((a.unsqueeze(2) - b.unsqueeze(1)) ** 2).sum(-1)  # [nA*nBc, T, T]
        if kind == "dtw":
            val, _ = soft_dtw(C.requires_grad_(True), gamma=gamma)
            val = val.detach() / T
        else:
            P = log_sinkhorn(C, eps=eps, iters=iters)
            val = (P * C).sum((-2, -1))
        val = val.view(nA, Bc.shape[0])
        best = torch.minimum(best, val.min(dim=1).values)
    return float(best.mean().cpu())


def per_dim_w1(X, Y, mu, sd):
    """Mean over dimensions of W1 between standardised marginals."""
    X = ((X - mu) / sd).reshape(-1, X.shape[-1]).numpy()
    Y = ((Y - mu) / sd).reshape(-1, Y.shape[-1]).numpy()
    n = min(len(X), len(Y), 20000)
    out = []
    for d in range(X.shape[-1]):
        xs = np.sort(np.random.default_rng(0).choice(X[:, d], n, replace=False))
        ys = np.sort(np.random.default_rng(1).choice(Y[:, d], n, replace=False))
        out.append(np.abs(xs - ys).mean())
    return float(np.mean(out))


def flatten_valid(obs, lengths):
    return torch.cat([obs[i, : int(L)] for i, L in enumerate(lengths)])


def evaluate_run(env, rollout_path, expert, expert_lengths, args, rng):
    """Metrics for one run, averaged over `args.n_splits` random expert splits.

    A single split is not enough: with ~10 trajectories per split the ratio
    varies so much that genuine expert data scored 2.90 against a 1.00 ideal.
    Averaging 8 random splits over all 128 demonstrations brings the
    calibration to 1.01 +- 0.03, which is what makes a learner number mean
    anything. `--self-test` reports the calibration at whatever settings are
    in force; read every number against it, not against 1.
    """
    lo, hi = PERIOD_RANGE[env]
    obs, lengths = load_eval_rollouts(rollout_path, args.max_episodes)

    n = expert.shape[0]
    acc = defaultdict(list)
    for split in range(args.n_splits):
        srng = np.random.default_rng(1000 + split)
        perm = srng.permutation(n)
        k = n // 3
        iR, iH = perm[:k], perm[k : 2 * k]
        ref, ref_len = expert[iR], expert_lengths[iR]
        hold, hold_len = expert[iH], expert_lengths[iH]
        flat_ref = flatten_valid(ref, ref_len)
        mu, sd = flat_ref.mean(0), flat_ref.std(0) + 1e-6
        for key, val in _one_split(obs, lengths, ref, ref_len, hold, hold_len,
                                   mu, sd, args, srng).items():
            acc[key].append(val)

    res = {k: float(np.mean(v)) for k, v in acc.items() if v}
    res.update({f"{k}_std": float(np.std(v)) for k, v in acc.items() if len(v) > 1})
    if not res:
        res = {"status": f"episodes shorter than window ({int(lengths.max())} < {args.window})"}
    try:
        p_learn, raw_l, mins_l = estimate_period_fundamental(obs, lengths, lo, hi, args.burn_in)
        p_exp, raw_e, _ = estimate_period_fundamental(expert, expert_lengths, lo, hi, args.burn_in)
        res["period_learner"] = p_learn
        res["period_expert"] = p_exp
        res["period_ratio"] = p_learn / p_exp if p_exp else float("nan")
        # The deepest minimum is not always the fundamental and no tolerance
        # separates "real multi-stride pattern" from "deep harmonic" honestly,
        # so carry the minima and let the reader judge an odd ratio.
        res["period_minima"] = ";".join(f"{l}:{sc:.3f}" for l, sc in mins_l[:4])
    except Exception as exc:
        res["period_error"] = str(exc)[:60]
    res["n_episodes"] = int(obs.shape[0])
    return res


def _one_split(obs, lengths, ref, ref_len, hold, hold_len, mu, sd, args, rng):
    dev = getattr(args, "device", "cpu")
    out = {}
    W = args.window
    w_learn = _windows(obs, lengths, W, args.n_windows, rng)
    w_hold = _windows(hold, hold_len, W, args.n_windows, rng)
    w_ref = _windows(ref, ref_len, W, args.n_windows, rng)
    if w_learn is None:
        return out
    if w_hold is not None and w_ref is not None:
        kinds = ("dtw", "ot") if getattr(args, "with_ot", False) else ("dtw",)
        for kind in kinds:
            # precision: how close is each learner window to *some* held-out
            # expert window. Low on its own is not proof of fidelity -- a
            # policy that repeats one stereotyped cycle scores well here.
            d_learn = alignment_distance(w_learn, w_hold, mu, sd, kind=kind, device=dev)
            d_exp = alignment_distance(w_ref, w_hold, mu, sd, kind=kind, device=dev)
            # coverage: how close is each held-out *expert* window to some
            # learner window. A mode-collapsed learner scores badly here, which
            # is what distinguishes it from a faithful one.
            c_learn = alignment_distance(w_hold, w_learn, mu, sd, kind=kind, device=dev)
            c_exp = alignment_distance(w_hold, w_ref, mu, sd, kind=kind, device=dev)
            out[f"{kind}_precision"] = d_learn / d_exp if d_exp > 0 else float("nan")
            out[f"{kind}_coverage"] = c_learn / c_exp if c_exp > 0 else float("nan")

    out["w1_per_dim"] = per_dim_w1(flatten_valid(obs, lengths), flatten_valid(hold, hold_len), mu, sd)
    return out


def self_test(env="hopper", window=32, n_windows=24):
    """Calibrate the metric on constructions with known answers.

    Run this before trusting any number: it reports what the metric says for
    genuine expert data (should sit near 1.0), for a deliberately
    mode-collapsed policy, for noise-corrupted motion, and for time-reversed
    motion. The expert-vs-expert row is the calibration baseline -- with
    finitely many demonstrations it is not exactly 1.0, and learner numbers
    should be read against it rather than against 1.
    """
    rng = np.random.default_rng(0)
    exp, elen = load_expert(env, 48)
    n = exp.shape[0] // 3
    A, al = exp[:n], elen[:n]
    B, bl = exp[n : 2 * n], elen[n : 2 * n]
    H, hl = exp[2 * n :], elen[2 * n :]
    fr = flatten_valid(B, bl)
    mu, sd = fr.mean(0), fr.std(0) + 1e-6
    wA = _windows(A, al, window, n_windows, rng)
    wB = _windows(B, bl, window, n_windows, rng)
    wH = _windows(H, hl, window, n_windows, rng)

    def score(wX):
        out = {}
        kinds = ("dtw", "ot") if getattr(args, "with_ot", False) else ("dtw",)
        for kind in kinds:
            out[f"{kind}_precision"] = (alignment_distance(wX, wH, mu, sd, kind=kind)
                                        / alignment_distance(wB, wH, mu, sd, kind=kind))
            out[f"{kind}_coverage"] = (alignment_distance(wH, wX, mu, sd, kind=kind)
                                       / alignment_distance(wH, wB, mu, sd, kind=kind))
        return out

    cases = [
        ("expert held-in (calibration)", wA),
        ("mode collapse: 1 window repeated", wA[:1].repeat(wA.shape[0], 1, 1)),
        ("expert + 0.5sd noise", wA + 0.5 * torch.randn_like(wA) * sd),
        ("expert + 2.0sd noise", wA + 2.0 * torch.randn_like(wA) * sd),
        ("expert reversed in time", torch.flip(wA, dims=[1])),
    ]
    print(f"self-test on {env} (window={window}, n_windows={n_windows})\n")
    print(f"  {'case':36s} {'dtw prec':>9s} {'dtw cov':>8s} {'ot prec':>8s} {'ot cov':>7s}")
    rows = []
    for tag, w in cases:
        r = score(w)
        rows.append(dict(case=tag, **r))
        print(f"  {tag:36s} {r['dtw_precision']:9.2f} {r['dtw_coverage']:8.2f} "
              f"{r['ot_precision']:8.2f} {r['ot_coverage']:7.2f}")
    print("\n  Expected: row 1 near 1.0; mode collapse good on precision but bad on")
    print("  coverage; noise degrades both; time reversal caught by dtw, NOT by ot.")
    return rows


# -------------------------------------------------------------------- main ---

def discover(env_filter=None, subdir="demos_at_budget"):
    """{env: {method: [rollout paths]}} from grouped_exp_logdirs.yaml.

    Defaults to `demos_at_budget/`, written by plotter/rollout_at_budget.py.
    The run's own `demos/` directory holds end-of-training rollouts, which for
    the reported runs are far past the budget the paper reports and therefore
    describe different policies.
    """
    g = yaml.safe_load(open(PLOTTER_DIR / "grouped_exp_logdirs.yaml"))
    found = defaultdict(lambda: defaultdict(list))
    for env, methods in g.items():
        if env_filter and env != env_filter:
            continue
        for m, runs in methods.items():
            for r in runs:
                pat = os.path.join("plotter", r["logdir"], subdir, "*.pt")
                for p in glob.glob(str(REPO / pat)):
                    found[env][m].append(p)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--n-windows", type=int, default=256)
    ap.add_argument("--max-episodes", type=int, default=32)
    ap.add_argument("--device", default="cpu", help="cuda:0 strongly recommended")
    ap.add_argument("--with-ot", action="store_true",
                    help="also report the OT distance. Off by default: it cannot "
                         "distinguish forward from backward motion (--self-test "
                         "shows 1.24 for a time-reversed expert vs 18.05 for "
                         "soft-DTW), so it is not a fidelity measure.")
    ap.add_argument("--n-splits", type=int, default=8,
                    help="random expert splits to average over; 1 is far too noisy")
    ap.add_argument("--expert-episodes", type=int, default=128)
    ap.add_argument("--burn-in", type=int, default=50)
    ap.add_argument("--out", default=str(OUT_DIR / "fidelity_metrics"))
    ap.add_argument("--subdir", default="demos_at_budget",
                    help="demos_at_budget (budget-matched) | demos (end of training)")
    ap.add_argument("--self-test", action="store_true",
                    help="calibrate on constructions with known answers and exit")
    args = ap.parse_args()

    if args.self_test:
        rows = self_test(args.env or "hopper", args.window, args.n_windows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "fidelity_self_test.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {OUT_DIR / 'fidelity_self_test.csv'}")
        return

    rng = np.random.default_rng(0)
    found = discover(args.env, args.subdir)
    if not found:
        print("No eval rollouts found locally. They live at <logdir>/demos/*.pt;")
        print("sync those directories from the cluster for the other environments.")
        return

    rows = []
    for env in found:
        expert, elen = load_expert(env, args.expert_episodes)
        print(f"\n### {env}  (expert {expert.shape[0]} episodes)")
        for method, paths in sorted(found[env].items()):
            for p in sorted(paths):
                seed = Path(p).parent.parent.name
                try:
                    r = evaluate_run(env, p, expert, elen, args, rng)
                except Exception as exc:
                    print(f"  {method:16s} {seed:12s} FAILED: {exc}")
                    continue
                r.update(environment=env, method=method, run=seed)
                rows.append(r)
                if "status" in r:
                    print(f"  {method:16s} {seed:12s} {r['status']}")
                    continue
                print(f"  {method:16s} {seed:12s} "
                      f"period {r.get('period_learner','?')}/{r.get('period_expert','?')} "
                      f"(ratio {r.get('period_ratio', float('nan')):.2f})  "
                      f"dtw prec {r.get('dtw_precision', float('nan')):.2f}x "
                      f"cov {r.get('dtw_coverage', float('nan')):.2f}x  "
                      f"w1 {r.get('w1_per_dim', float('nan')):.3f}")
    if rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(args.out + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}.csv  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
