"""SHAC expert for the drone (compact: deterministic actor, TD(lambda) critic, horizon H, through DroneEnv's
differentiable Warp step). Saves the checkpoint with the best evaluation success to --out.

    PYTHONPATH=. uv run --locked python scripts/drone_x2b/train_expert.py --iters 1500 --out experts/demos/Warp_Drone_expert_shac.pt
"""

import argparse
import json
import time

import torch
import torch.nn as nn

from common import Actor, evaluate, mlp
from mineral.envs.drone import DEFAULT_TASK, HARD_TASK, DroneEnv

p = argparse.ArgumentParser()
p.add_argument('--task', default='default')
p.add_argument('--iters', type=int, default=2000)
p.add_argument('--n', type=int, default=512)
p.add_argument('--H', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--gamma', type=float, default=0.99)
p.add_argument('--lam', type=float, default=0.95)
p.add_argument('--noise', type=float, default=0.0)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', required=True)
p.add_argument('--init', default='')
args = p.parse_args()
torch.manual_seed(args.seed)
dev = 'cuda:0'
task = DEFAULT_TASK if args.task == 'default' else HARD_TASK


if __name__ == '__main__':
    env = DroneEnv(args.n, dev, seed=args.seed, no_grad=False, **task)
    actor, critic = Actor().to(dev), mlp(18, 1).to(dev)
    if args.init:
        actor.load_state_dict(torch.load(args.init, weights_only=False)['actor'])
    critic_t = mlp(18, 1).to(dev)
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr, betas=(0.7, 0.95))
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.7, 0.95))
    # spread episode phases
    env.progress_buf[:] = torch.randint(0, env.episode_length, (args.n,), device=dev)
    o = env.obs()
    H, g, lam = args.H, args.gamma, args.lam
    t0 = time.time()
    log = []
    best = -1
    for it in range(args.iters):
        frac = 1 - it / args.iters
        for pg in opt_a.param_groups + opt_c.param_groups:
            pg['lr'] = args.lr * frac
        env.clear_grad()
        o = env.obs()
        obs_buf, rew_buf, done_buf, nextv_obs = [], [], [], []
        loss = 0.0
        disc = torch.ones(args.n, device=dev)
        for t in range(H):
            a = actor(o)
            if args.noise > 0:
                a = a + args.noise * torch.randn_like(a)
            obs_buf.append(o.detach())
            o2, r, d, info = env.step(a)
            r = torch.nan_to_num(r, nan=0.0)
            loss = loss - (disc * r).sum()
            ob = info['obs_before_reset']
            term = info['termination']
            # bootstrap at episode ends (truncation) and at the window end
            if t == H - 1:
                v = critic_t(torch.nan_to_num(ob)).squeeze(-1)
                loss = loss - (disc * g * v * (~term).float()).sum()
            elif d.any():
                v = critic_t(torch.nan_to_num(ob)).squeeze(-1)
                loss = loss - (disc * g * v * (d & ~term).float()).sum()
            disc = torch.where(d, torch.ones_like(disc), disc * g)
            rew_buf.append(r.detach())
            done_buf.append(d.clone())
            nextv_obs.append((torch.nan_to_num(ob.detach()), term.clone()))
            o = o2['obs']
        loss = loss / (H * args.n)
        opt_a.zero_grad()
        loss.backward()
        gn = nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        if torch.isfinite(gn):
            opt_a.step()
        # critic targets: TD(lambda)
        with torch.no_grad():
            vnext = torch.stack([critic_t(ob).squeeze(-1) * (~tm).float() for ob, tm in nextv_obs])  # [H, n]
            R = torch.stack(rew_buf)
            D = torch.stack(done_buf).float()
            tgt = torch.zeros_like(R)
            nxt = vnext[-1]
            for t in reversed(range(H)):
                # lambda-return: r + g((1-lam) V(s') + lam G_{t+1}); at episode end G_{t+1} := V(s')
                if t == H - 1:
                    G = R[t] + g * vnext[t]
                else:
                    G = R[t] + g * ((1 - lam) * vnext[t] + lam * (D[t] * vnext[t] + (1 - D[t]) * nxt))
                tgt[t] = G
                nxt = G
            X = torch.cat(obs_buf)
            Y = tgt.reshape(-1)
        for _ in range(16):
            for idx in torch.randperm(len(X), device=dev).chunk(4):
                lc = ((critic(X[idx]).squeeze(-1) - Y[idx]) ** 2).mean()
                opt_c.zero_grad()
                lc.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_c.step()
        with torch.no_grad():
            for pt, pc in zip(critic_t.parameters(), critic.parameters()):
                pt.lerp_(pc, 0.2)
        if it % 100 == 0 or it == args.iters - 1:
            ev = evaluate(actor, task)
            ev.update(it=it, loss=loss.item(), closs=lc.item(), gn=gn.item(), t=time.time() - t0)
            print(json.dumps(ev), flush=True)
            log.append(ev)
            if ev['success'] >= best:
                best = ev['success']
                torch.save({'actor': actor.state_dict(), 'task': task, 'eval': ev}, args.out)
    torch.save({'actor': actor.state_dict(), 'task': task, 'eval': ev, 'log': log}, args.out.replace('.pt', '_final.pt'))
