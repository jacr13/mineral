"""Goal-matched mentor test (ICML plan X2b): demos on the absolute-frame drone with a finite set of targets.

The observation is [position, target, R, v, w] (21-d), so demonstrations for different targets are different tasks.
Targets: n_targets points uniform in [-target_range, target_range]^3 (fixed seed); the first n_seen are demonstrated,
the rest are held out. Demo i flies to seen target i % n_seen, so the first k * n_seen demos hold k per target.
The expert (trained on the relative frame) reads the relative observation.
"""

import argparse
import json

import torch
from common import Actor

from mineral.envs.drone import DroneEnv, rel_from_abs

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--n', type=int, default=128)
p.add_argument('--n_targets', type=int, default=12)
p.add_argument('--n_seen', type=int, default=8)
p.add_argument('--target_seed', type=int, default=7)
p.add_argument('--seed', type=int, default=1000)
p.add_argument('--out_dir', default='experts/demos')
p.add_argument('--name', default='Warp_DroneAbs')
args = p.parse_args()
dev = 'cuda:0'
ck = torch.load(args.ckpt, weights_only=False)
actor = Actor().to(dev)
actor.load_state_dict(ck['actor'])
g = torch.Generator().manual_seed(args.target_seed)
r = float(ck['task'].get('target_range', 2.0))
targets = ((torch.rand(args.n_targets, 3, generator=g) * 2 - 1) * r).tolist()
seen = list(range(args.n_seen))
task = dict(ck['task'], obs_frame='absolute', targets=targets, target_ids=seen)
N = args.n
env = DroneEnv(N, dev, seed=args.seed, no_grad=True, **task)
# demo i flies to seen target i % n_seen from a start offset drawn as in the default task
q, qd, wps = env.sample_init(N)
tgt = torch.tensor(targets, device=dev)[torch.tensor(seen, device=dev)[torch.arange(N, device=dev) % len(seen)]]
q[:, 0:3] = q[:, 0:3] - wps[:, 0] + tgt
env.set_state(torch.arange(N, device=dev), q, qd, tgt)
T = env.episode_length
O, O1, A, R, D = (torch.zeros(N, T, k, device=dev) for k in (21, 21, 4, 1, 1))
done_any = torch.zeros(N, dtype=torch.bool, device=dev)
lengths = torch.full((N,), T, dtype=torch.long, device=dev)
succ = torch.zeros(N, dtype=torch.bool, device=dev)
o = env.obs()
with torch.no_grad():
    for t in range(T):
        a = actor(rel_from_abs(o))
        o2, rew, d, info = env.step(a)
        live = ~done_any
        O[live, t], A[live, t], R[live, t, 0] = o[live], a[live].clamp(-1, 1), rew[live]
        O1[live, t] = info['obs_before_reset'][live]
        newly = d & live
        D[newly, t, 0] = 1
        lengths[newly] = t + 1
        succ |= newly & info['successes']
        done_any |= d
        o = o2['obs']
ret = R[..., 0].sum(1)
print(f'success {succ.float().mean():.3f}, return {ret.mean():.1f} +- {ret.std():.1f}, lengths min {lengths.min()}')
out = {'obs': {'obs': O.cpu()}, 'act': A.cpu(), 'next_obs': {'obs': O1.cpu()}, 'rew': R[..., 0].cpu(),
       'done': D[..., 0].long().cpu(), 'lengths': lengths.cpu()}  # fmt: skip
path = f'{args.out_dir}/{args.name}_demos{N}_return{ret.mean():.0f}_len{T}.pt'
torch.save(out, path)
meta = {'targets': targets, 'seen': seen, 'held_out': list(range(args.n_seen, args.n_targets)),
        'success': succ.float().mean().item(), 'return': ret.mean().item(), 'task': task, 'ckpt': args.ckpt}  # fmt: skip
json.dump(meta, open(path.replace('.pt', '_meta.json'), 'w'), indent=1)
print(path)
