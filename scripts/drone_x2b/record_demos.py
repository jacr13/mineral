"""Record expert demos in the experts/demos format (obs/next_obs {'obs': [N,T,18]}, act, rew, done, lengths)."""

import argparse

import torch
from common import Actor

from mineral.envs.drone import DroneEnv

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--n', type=int, default=128)
p.add_argument('--seed', type=int, default=1000)
p.add_argument('--out_dir', default='experts/demos')
p.add_argument('--name', default='Warp_Drone')
p.add_argument('--meta_dir', default='experts/demos')
args = p.parse_args()
dev = 'cuda:0'
ck = torch.load(args.ckpt, weights_only=False)
actor = Actor().to(dev)
actor.load_state_dict(ck['actor'])
task = ck['task']
env = DroneEnv(args.n, dev, seed=args.seed, no_grad=True, **task)
T = env.episode_length
N = args.n
O, O1, A, R, D = (torch.zeros(N, T, k, device=dev) for k in (18, 18, 4, 1, 1))
done_any = torch.zeros(N, dtype=torch.bool, device=dev)
lengths = torch.full((N,), T, dtype=torch.long, device=dev)
succ = torch.zeros(N, dtype=torch.bool, device=dev)
o = env.reset()['obs']
with torch.no_grad():
    for t in range(T):
        a = actor(o)
        o2, r, d, info = env.step(a)
        live = ~done_any
        O[live, t], A[live, t], R[live, t, 0] = o[live], a[live].clamp(-1, 1), r[live]
        O1[live, t] = info['obs_before_reset'][live]  # relative to the target of this step (pre-switch, pre-reset)
        newly = d & live
        D[newly, t, 0] = 1
        lengths[newly] = t + 1
        succ |= newly & info['successes']
        done_any |= d
        o = o2['obs']
ret = R[..., 0].sum(1)
print(f'success {succ.float().mean():.3f}, return {ret.mean():.1f} +- {ret.std():.1f}, lengths min {lengths.min()}')
out = {
    'obs': {'obs': O.cpu()},
    'act': A.cpu(),
    'next_obs': {'obs': O1.cpu()},
    'rew': R[..., 0].cpu(),
    'done': D[..., 0].long().cpu(),
    'lengths': lengths.cpu(),
}
path = f'{args.out_dir}/{args.name}_demos{N}_return{ret.mean():.0f}_len{T}.pt'
torch.save(out, path)
torch.save({'success': succ.cpu(), 'return': ret.cpu(), 'task': task, 'seed': args.seed, 'ckpt': args.ckpt},
           args.meta_dir + '/' + path.split('/')[-1].replace('.pt', '_meta.pt'))  # fmt: skip
print(path)
