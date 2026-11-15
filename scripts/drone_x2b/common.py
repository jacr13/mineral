"""Shared pieces of the drone feasibility scripts (policy network, evaluation)."""

import torch
import torch.nn as nn

from mineral.envs.drone import DroneEnv

dev = 'cuda:0'


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(18, 4)
        with torch.no_grad():  # start near hover (a = -0.31)
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.fill_(-0.32)

    def forward(self, o):
        return torch.tanh(self.net(o))


@torch.no_grad()
def evaluate(actor, task, n=256, seed=123):
    """One episode per env from seeded initial states: mean return, success rate, median final distance."""
    env = DroneEnv(n, dev, seed=seed, no_grad=True, **task)
    o = env.reset()['obs']
    ret = torch.zeros(n, device=dev)
    done_any = torch.zeros(n, dtype=torch.bool, device=dev)
    succ = torch.zeros(n, dtype=torch.bool, device=dev)
    final_d = torch.zeros(n, device=dev)
    for t in range(env.episode_length):
        o, r, d, info = env.step(actor(o))
        o = o['obs']
        ret += torch.where(done_any, 0.0, r)
        newly = d & ~done_any
        succ |= newly & info['successes']
        final_d = torch.where(newly, info['dist'], final_d)
        done_any |= d
    return dict(ret=ret.mean().item(), ret_std=ret.std().item(), success=succ.float().mean().item(),
                final_d_median=final_d.median().item())  # fmt: skip
