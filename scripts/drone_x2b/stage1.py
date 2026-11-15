"""Stage I check: invert one drone step per demo transition with MENTOR's lm_invert (state-only).

Reports the relative residual ||w (step(s, a) - s1)|| / ||w (s1 - s)|| (w = 1 / std(s1 - s), as MENTOR) for the
recovered actions and, as an AUDIT reference, for the stored expert actions; saves the recovered actions.
"""

import argparse
import time

import torch

from mineral.agents.diffil.mentor import lm_invert
from mineral.agents.diffil.mentor_sim import SimCounter
from mineral.envs.drone import DroneOneStep

p = argparse.ArgumentParser()
p.add_argument('--demos', required=True)
p.add_argument('--num_demos', type=int, default=128)
p.add_argument('--iters', type=int, default=5)
p.add_argument('--jacobian', default='fd')
p.add_argument('--batch', type=int, default=9600)
p.add_argument('--out', required=True)
args = p.parse_args()
dev = 'cuda:0'
d = torch.load(args.demos, weights_only=False)
N = min(args.num_demos, d['act'].shape[0])
L = d['lengths'][:N]
T = d['act'].shape[1]
valid = torch.arange(T)[None] < L[:, None]
S = d['obs']['obs'][:N][valid].to(dev)
S1 = d['next_obs']['obs'][:N][valid].to(dev)
A_true = d['act'][:N][valid].to(dev)  # AUDIT only: replay residual of the stored actions
w = 1.0 / (S1 - S).std(0).clamp_min(1e-8)
counter = SimCounter()
B = args.batch
step = DroneOneStep(B, dev, counter, requires_grad=(args.jacobian == 'autodiff'))


def chunks(n):
    for i in range(0, n, B):
        yield i, min(i + B, n)


def pad(x):
    k = x.shape[0]
    return torch.cat([x, x[-1:].expand(B - k, *x.shape[1:])]) if k < B else x


rel_true, rel_rec, acts, dens = [], [], [], []
t0 = time.time()
for i, j in chunks(len(S)):
    s, s1, at = pad(S[i:j]), pad(S1[i:j]), pad(A_true[i:j])
    k = j - i
    denom = ((s1 - s) * w).norm(dim=-1).clamp_min(1e-9)
    with torch.no_grad():
        rel_true.append((((step(s, at, grad=False) - s1) * w).norm(dim=-1) / denom)[:k])
    a, c = lm_invert(step, s, s1, w, iters=args.iters, jacobian=args.jacobian)
    rel_rec.append((c.sqrt() / denom)[:k])
    acts.append(a[:k])
    dens.append(denom[:k])
el = time.time() - t0
rel_true, rel_rec, acts, dens = torch.cat(rel_true), torch.cat(rel_rec), torch.cat(acts), torch.cat(dens)
qs = torch.tensor([0.5, 0.95, 0.99], device=dev)
print(f'{len(S)} transitions, {counter.total / len(S):.1f} sim steps per transition, {el:.1f}s')
print('stored actions (AUDIT) rel residual q50/95/99/max', [f'{v:.2e}' for v in rel_true.quantile(qs).tolist()], f'{rel_true.max():.2e}')
print('recovered actions      rel residual q50/95/99/max', [f'{v:.2e}' for v in rel_rec.quantile(qs).tolist()], f'{rel_rec.max():.2e}')
for name, rel in (('stored (AUDIT)', rel_true), ('recovered', rel_rec)):
    ab = rel * dens
    mv = dens > 0.1
    print(f'{name:15s} absolute weighted residual q50/95/99 {[f"{v:.2e}" for v in ab.quantile(qs).tolist()]}; '
          f'rel. residual on moving transitions (||w ds|| > 0.1, {mv.float().mean():.2f} of all) q50/95/99 '
          f'{[f"{v:.2e}" for v in rel[mv].quantile(qs).tolist()]}')  # fmt: skip
err = (acts - A_true).abs()
print(f'AUDIT |a_rec - a_true|: mean {err.mean():.2e}, q99 {err.max(-1).values.quantile(0.99):.2e}, max {err.max():.2e}; '
      f'frac stored |a| > 0.999: {(A_true.abs() > 0.999).float().mean():.3f}')  # fmt: skip
A_rec = torch.zeros_like(d['act'][:N])
A_rec[valid] = acts.cpu()
torch.save({'act': A_rec, 'rel_rec': rel_rec.cpu(), 'rel_true': rel_true.cpu(), 'num_demos': N}, args.out)
