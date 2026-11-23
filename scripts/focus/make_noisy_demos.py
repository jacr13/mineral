"""Noisy copies of the demonstration files (robustness to imperfect demonstrations, Joao 2026-11-23).

usage: python scripts/focus/make_noisy_demos.py OUT_DIR SIGMA DEMO_FILE ...   (env KEEP: episodes kept, default 8)

Every observation of every demonstration gets independent Gaussian noise with standard deviation SIGMA times that
dimension's standard deviation over the valid steps of the whole 128-episode pool. The noise is drawn once with a
fixed seed, so every method and run sees the same noisy demonstrations. next_obs[t] receives the same noise as
obs[t + 1] (they are the same state), and the last next_obs of each episode gets its own draw. Actions, rewards and
done flags are unchanged (the methods under test use observations only; the reference rows BC / GAIL see clean
actions). Only the first KEEP episodes are written (every method uses n_trajs = 8; the scales are computed on the
whole pool first), so the file shipped with each run stays small. Output: OUT_DIR/<name>_noise<SIGMA>.pt, plus a
.json with the per-dimension scales.
"""
import json
import os
import sys

import torch


def noisy(path, sigma, out_dir, seed=0):
    d = torch.load(path, map_location='cpu', weights_only=False)
    g = torch.Generator().manual_seed(seed)
    lengths = d['lengths']
    out = dict(d)
    out['obs'], out['next_obs'], scales = {}, {}, {}
    for k, o in d['obs'].items():
        n, T = o.shape[:2]
        valid = torch.arange(T)[None, :] < lengths[:, None]
        std = o[valid].float().std(0)
        eps = torch.randn((n, T + 1) + o.shape[2:], generator=g) * (sigma * std)
        out['obs'][k] = o + eps[:, :T].to(o.dtype)
        out['next_obs'][k] = d['next_obs'][k] + eps[:, 1:T + 1].to(o.dtype)
        scales[k] = std.tolist()
    keep = int(os.environ.get('KEEP', 8))
    for k in ('act', 'rew', 'done', 'lengths'):
        if k in out:
            out[k] = out[k][:keep].clone()  # clone: a saved slice would keep the whole storage
    out['obs'] = {k: v[:keep].clone() for k, v in out['obs'].items()}
    out['next_obs'] = {k: v[:keep].clone() for k, v in out['next_obs'].items()}
    base = os.path.basename(path)[:-3]
    dst = os.path.join(out_dir, f'{base}_noise{sigma:g}.pt')
    torch.save(out, dst)
    json.dump(dict(source=os.path.basename(path), sigma=sigma, seed=seed, episodes_kept=keep, per_dim_std=scales),
              open(dst[:-3] + '.json', 'w'), indent=1)
    return dst


if __name__ == '__main__':
    out_dir, sigma = sys.argv[1], float(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)
    for p in sys.argv[3:]:
        print(noisy(p, sigma, out_dir))
