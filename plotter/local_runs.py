"""Read run metrics straight off disk, without W&B.

Each run directory written by mineral carries the full training curve as two
numpy arrays -- ``my_ep_steps_hist.npy`` (agent steps) and
``my_ep_rewards_hist.npy`` (episode return) -- plus its fully resolved config.
That is everything the rebuttal analysis takes from the W&B API, at ~80 KB per
run, so a run can be scored from a directory alone.

Validated against the API on the six reported Hopper FOCUS-OT-L2 runs: four
agree exactly and two differ by <0.09 on ~4780 (2e-5 relative), which is W&B's
history sampling rather than a difference in the data.

Use this when the cluster cannot reach W&B. Syncing an offline run needs a
working upload path; copying two .npy files and a yaml does not.
"""

from pathlib import Path

import numpy as np
import yaml

STEPS = "my_ep_steps_hist.npy"
REWARDS = "my_ep_rewards_hist.npy"


def read_run(logdir):
    """-> dict(steps, rewards, config, logdir) for one run directory."""
    d = Path(logdir)
    steps = np.load(d / STEPS)
    rewards = np.load(d / REWARDS)
    if len(steps) != len(rewards):
        n = min(len(steps), len(rewards))
        steps, rewards = steps[:n], rewards[:n]
    cfg = {}
    p = d / "resolved_config.yaml"
    if p.exists():
        cfg = yaml.safe_load(p.read_text()) or {}
    return {"steps": steps, "rewards": rewards, "config": cfg, "logdir": str(d)}


def final_at_budget(run, cap, tail=5, min_coverage=0.98):
    """Tail-mean return at ``cap`` steps, or None if the run stopped short.

    Mirrors plotter/rebuttal_plots.final_at_budget, including the coverage
    guard: a run that exits early on the JobClock must not be scored as if it
    had reached the budget.
    """
    steps, rewards = run["steps"], run["rewards"]
    m = steps <= cap
    if not m.any():
        return None, 0.0
    reached = float(steps[m].max())
    if reached < min_coverage * cap:
        return None, reached
    return float(rewards[m][-tail:].mean()), reached


def dig(cfg, *path, default=None):
    cur = cfg
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def discover(root, pattern="**/my_ep_steps_hist.npy"):
    """Every run directory under ``root`` that has a curve on disk."""
    return sorted(p.parent for p in Path(root).glob(pattern))


# --------------------------------------------------------------- tb events ---
# The .npy files carry only returns/lengths/steps. The TensorBoard event file in
# <logdir>/tb/ carries every logged scalar -- 77 tags, including
# train_stats/Lk_min and Lk_mean (the inputs to the K_eff measurement),
# imitation_loss, pseudo_reward_mean/std (which show whether the reward scale
# moved, the failure mode of the unbalanced-Sinkhorn underflow bug) and
# eval_scores/* as a curve rather than the single snapshot in scores.json.
#
# Event files are ~73 MB per run, so copy them for the handful of runs that need
# diagnostics rather than for a whole batch; the .npy files are enough to score
# return at budget.

def tb_tags(logdir):
    """Every scalar tag in a run's event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    f = _tb_file(logdir)
    ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
    ea.Reload()
    return sorted(ea.Tags()["scalars"])


def read_tb(logdir, tags):
    """-> {tag: (steps, values)} read straight from the event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    f = _tb_file(logdir)
    ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    out = {}
    for t in tags:
        if t not in have:
            continue
        ev = ea.Scalars(t)
        out[t] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def _tb_file(logdir):
    files = sorted(Path(logdir).glob("tb/events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"no event file under {logdir}/tb")
    return max(files, key=lambda p: p.stat().st_size)
