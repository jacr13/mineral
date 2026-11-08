"""Convert a run's saved evaluation rollouts into the expert-demo format.

A `run=train_eval` run writes `<logdir>/demos/*.pt` as a dict keyed by episode
index:  {0: {"obs": {...}, "act": ..., "next_obs": {...}, "rew": ..., "done": ...}, 1: {...}}

`mineral.common.demos.get_demos` expects the stacked format instead:
    {"obs": {k: [N, T, ...]}, "act": [N, T, A], "next_obs": {...},
     "rew": [N, T], "done": [N, T], "lengths": [N]}

Copying the rollout file straight into experts/demos/ therefore fails with
`KeyError: 'obs'`. experts/experts.py does this conversion but reprocesses
every file in experts/dflex/ and derives its own output name; this does one
file to one destination, which is what adding a new task needs.

    python experts/convert_rollouts.py <rollout.pt> <out.pt>
"""

import argparse
from pathlib import Path

import torch


def convert(src, dst=None):
    data = torch.load(src, map_location="cpu")
    if not (isinstance(data, dict) and 0 in data):
        raise ValueError(f"{src}: not a per-episode rollout file (keys: {list(data)[:5]})")

    eps = [data[k] for k in sorted(data)]
    max_t = max(int(e["act"].shape[0]) for e in eps)

    obs_keys = list(eps[0]["obs"].keys())
    out_obs = {k: [] for k in obs_keys}
    out_next = {k: [] for k in obs_keys}
    out_act, out_rew, out_done, lengths = [], [], [], []

    def pad(v, fill=0.0):
        p = torch.full((max_t, *v.shape[1:]), fill, dtype=v.dtype)
        p[: v.shape[0]] = v
        return p

    for e in eps:
        t = int(e["act"].shape[0])
        lengths.append(t)
        for k in obs_keys:
            out_obs[k].append(pad(e["obs"][k]))
            out_next[k].append(pad(e["next_obs"][k]))
        out_act.append(pad(e["act"]))
        out_rew.append(pad(e["rew"]))
        # pad `done` with 1 so padded tail counts as terminated, matching
        # experts/experts.py
        out_done.append(pad(e["done"], fill=1))

    bundle = {
        "obs": {k: torch.stack(v) for k, v in out_obs.items()},
        "act": torch.stack(out_act),
        "next_obs": {k: torch.stack(v) for k, v in out_next.items()},
        "rew": torch.stack(out_rew),
        "done": torch.stack(out_done),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }

    n = bundle["rew"].shape[0]
    ret = float(bundle["rew"].sum(-1).mean())
    avg_len = int(bundle["lengths"].float().mean())
    if dst is None:
        stem = "_".join(Path(src).name.split("_")[:2])
        dst = f"experts/demos/{stem}_demos{n}_return{int(ret)}_len{avg_len}.pt"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, dst)
    print(f"  episodes      : {n}")
    print(f"  obs           : { {k: tuple(v.shape) for k, v in bundle['obs'].items()} }")
    print(f"  act           : {tuple(bundle['act'].shape)}")
    print(f"  mean return   : {ret:.1f}")
    print(f"  mean length   : {avg_len}")
    print(f"  wrote         : {dst}")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None)
    a = ap.parse_args()
    convert(a.src, a.dst)
