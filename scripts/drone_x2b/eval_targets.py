"""Goal-matched mentor test: evaluate a MENTOR drone_abs policy (final.pth) on seen and held-out targets.

One episode per env from seeded starts; success = within the success radius for the last hold_steps steps.
usage: python eval_targets.py --ckpt <run>/ckpt/final.pth --meta experts/demos/Warp_DroneAbs_..._meta.json
"""

import argparse
import json

import torch

from mineral.agents.diffil.mentor import MentorPolicy
from mineral.envs.drone import DroneEnv

p = argparse.ArgumentParser()
p.add_argument('--ckpt', nargs='+', required=True)
p.add_argument('--meta', required=True)
p.add_argument('--n', type=int, default=256)
p.add_argument('--seed', type=int, default=4242)
args = p.parse_args()
dev = 'cuda:0'
meta = json.load(open(args.meta))


@torch.no_grad()
def evaluate(pi, ids):
    task = dict(meta['task'], target_ids=ids)
    env = DroneEnv(args.n, dev, seed=args.seed, no_grad=True, **task)
    o = env.reset()['obs']
    ret = torch.zeros(args.n, device=dev)
    done = torch.zeros(args.n, dtype=torch.bool, device=dev)
    succ = torch.zeros(args.n, dtype=torch.bool, device=dev)
    for _ in range(env.episode_length):
        o, r, d, info = env.step(pi(o))
        o = o['obs']
        ret += torch.where(done, 0.0, r)
        succ |= d & ~done & info['successes']
        done |= d
    return ret.mean().item(), succ.float().mean().item()


for path in args.ckpt:
    sd = torch.load(path, map_location=dev, weights_only=False)['policy']
    units = [sd[k].shape[0] for k in sorted(k for k in sd if k.startswith('net.') and k.endswith('.weight'))][:-1]
    pi = MentorPolicy(sd['mean'].shape[0], 4, sd['mean'], sd['std'], units=units).to(dev)
    pi.load_state_dict(sd)
    pi.eval()
    rs, ss = evaluate(pi, meta['seen'])
    rh, sh = evaluate(pi, meta['held_out'])
    print(f'{path}\tseen return {rs:.1f} success {ss:.3f}\theld-out return {rh:.1f} success {sh:.3f}')
