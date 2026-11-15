"""Cloning from k demos: BC with the stored expert actions (labelled reference) and with Stage I-recovered actions."""

import argparse
import json

import torch
import torch.nn as nn
from common import Actor, dev, evaluate


def to_yaw_frame(o):
    # rotate every vector by the inverse heading (yaw about y, from the body x axis projected on the ground plane)
    R = o[:, 3:12].view(-1, 3, 3)
    psi = torch.atan2(-R[:, 2, 0], R[:, 0, 0])
    c, s_ = torch.cos(psi), torch.sin(psi)
    z, one = torch.zeros_like(c), torch.ones_like(c)
    Ry = torch.stack([c, z, s_, z, one, z, -s_, z, c], -1).view(-1, 3, 3)  # rotation by psi about y
    Rt = Ry.transpose(1, 2)
    v = lambda x: (Rt @ x[:, :, None])[:, :, 0]
    return torch.cat([v(o[:, 0:3]), (Rt @ R).reshape(-1, 9), v(o[:, 12:15]), v(o[:, 15:18])], -1)


class Framed(nn.Module):
    def __init__(self, frame):
        super().__init__()
        self.pi, self.frame = Actor(), frame

    def forward(self, o):
        return self.pi(to_yaw_frame(o) if self.frame == 'yaw' else o)


p = argparse.ArgumentParser()
p.add_argument('--demos', required=True)
p.add_argument('--recovered', required=True)
p.add_argument('--expert', required=True)
p.add_argument('--k', type=int, default=8)
p.add_argument('--seeds', type=int, default=3)
p.add_argument('--steps', type=int, default=5000)
p.add_argument('--n_eval', type=int, default=256)
p.add_argument(
    '--frame', default='world', help="world | yaw: the cloned policy sees the observation in the drone's heading frame"
)
args = p.parse_args()
d = torch.load(args.demos, weights_only=False)
rec = torch.load(args.recovered, weights_only=False)
ck = torch.load(args.expert, weights_only=False)
task = ck['task']
expert = Actor().to(dev)
expert.load_state_dict(ck['actor'])
ev_exp = evaluate(expert, task, n=args.n_eval)
print('expert', json.dumps(ev_exp), flush=True)
k = args.k
T = d['act'].shape[1]
valid = torch.arange(T)[None] < d['lengths'][:k, None]
X = d['obs']['obs'][:k][valid].to(dev)
labels = {'bc_expert_actions(reference)': d['act'][:k][valid].to(dev), 'stage1_clone': rec['act'][:k][valid].to(dev)}
res = {}
for name, Y in labels.items():
    out = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        pi = Framed(args.frame).to(dev)
        opt = torch.optim.Adam(pi.parameters(), lr=1e-3)
        for it in range(args.steps):
            for g in opt.param_groups:
                g['lr'] = 1e-3 * (1 - it / args.steps)
            idx = torch.randint(0, len(X), (256,), device=dev)
            loss = ((pi(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        ev = evaluate(pi, task, n=args.n_eval)
        ev['train_mse'] = loss.item()
        out.append(ev)
        print(name, seed, json.dumps(ev), flush=True)
    r = torch.tensor([o['ret'] for o in out])
    s = torch.tensor([o['success'] for o in out])
    res[name] = dict(ret=r.mean().item(), ret_std=r.std().item(), ret_ratio=(r / ev_exp['ret']).mean().item(),
                     success=s.mean().item(), success_std=s.std().item(),
                     success_ratio=(s / max(ev_exp['success'], 1e-9)).mean().item())  # fmt: skip
print('SUMMARY', json.dumps({'k': k, 'frame': args.frame, 'expert': ev_exp, **res}))
