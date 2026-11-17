"""MENTOR's mentor on the NablaEngine G1 (X9) and the student policy.

The mentor is a receding-horizon planner towards time-indexed reference states; the student has NablaMimic's inputs
and encoder-decoder architecture.

The mentor never sees the NablaMimic reward: its cost is a weighted squared distance between simulated and reference
states (terminal state at f + K, plus w_run times the same distance at the intermediate steps), optionally a
setpoint-change term (w_rate) and an action-size term (w_act); Adam on the K residual actions through the engine,
best iterate kept.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .g1_track import G1Track


@dataclass
class MentorConfig:
    K: int = 16
    iters: int = 10
    lr: float = 0.05
    w_run: float = 0.1
    w_joint: float = 1.0
    w_jvel: float = 0.01
    w_z: float = 10.0
    w_xy: float = 1.0
    w_ori: float = 1.0
    w_rvel: float = 0.1
    w_act: float = 0.0
    w_rate: float = 0.0

    @classmethod
    def from_args(cls, args):
        return cls(**{k: getattr(args, k) for k in cls.__dataclass_fields__})


class Mentor:
    def __init__(self, env: G1Track, cfg: MentorConfig):
        self.env, self.cfg = env, cfg
        self.wj = torch.full((env.nu,), cfg.w_joint, device=env.device)

    def state_cost(self, q, v, f):
        c_, env = self.cfg, self.env
        r, rv = env.ref_qpos[f], env.ref_qvel[f]
        quat = q[:, 3:7] / q[:, 3:7].norm(dim=-1, keepdim=True)
        c = (self.wj * (q[:, 7:] - r[:, 7:]) ** 2).sum(-1)
        c = c + c_.w_jvel * ((v[:, 6:] - rv[:, 6:]) ** 2).sum(-1)
        c = c + c_.w_z * (q[:, 2] - r[:, 2]) ** 2 + c_.w_xy * ((q[:, :2] - r[:, :2]) ** 2).sum(-1)
        c = c + c_.w_ori * 4.0 * (1.0 - (quat * r[:, 3:7]).sum(-1) ** 2)
        c = c + c_.w_rvel * ((v[:, :6] - rv[:, :6]) ** 2).sum(-1)
        return c

    def plan_cost(self, q, v, w, f, delta, u_prev):
        """Total planning cost (B,) of the residual plan delta (B, K, nu) from state (q, v, w) at frame f.

        u_prev: the last applied setpoint (start of the setpoint-change term).
        """
        c_, env = self.cfg, self.env
        cost = 0.0
        for k in range(c_.K):
            fk = env.next_frame(f, k)
            u = env.ref_qpos[env.next_frame(fk), 7:] + env.action_scale * delta[:, k]
            if c_.w_rate:
                cost = cost + c_.w_rate * (((u - u_prev) / env.action_scale) ** 2).sum(-1)
                u_prev = u
            ctrl = torch.clamp(u, env.ctrl_lo, env.ctrl_hi)
            q, v, w = env.step(q, v, w, ctrl)
            wk = 1.0 if k == c_.K - 1 else c_.w_run
            if wk:
                cost = cost + wk * self.state_cost(q, v, env.next_frame(f, k + 1))
        if c_.w_act:
            cost = cost + c_.w_act * (delta ** 2).mean((1, 2))
        return cost

    def plan(self, q, v, w, f, delta, u_prev, iters=None):
        """Adam on the residual plan.

        Returns the best iterate per world (by its own cost), its cost and the cost of the initial plan.
        """
        iters = self.cfg.iters if iters is None else iters
        best = delta.clone()
        with torch.no_grad():
            best_cost = self.plan_cost(q, v, w, f, delta, u_prev)
        cost0 = best_cost.clone()
        x = delta.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=self.cfg.lr)
        for _ in range(iters):
            c = self.plan_cost(q, v, w, f, x, u_prev)
            cd = c.detach()
            better = cd < best_cost
            best[better], best_cost = x.detach()[better], torch.where(better, cd, best_cost)
            opt.zero_grad(set_to_none=True)
            c.sum().backward()
            x.grad.nan_to_num_(0.0, 0.0, 0.0)
            opt.step()
        if iters:
            with torch.no_grad():
                cd = self.plan_cost(q, v, w, f, x, u_prev)
            better = cd < best_cost
            best[better], best_cost = x.detach()[better], torch.where(better, cd, best_cost)
        return best, best_cost, cost0

    def sim_steps(self, B, iters=None):
        """(forward+backward, forward-only) simulator steps of one plan() call on B worlds."""
        iters = self.cfg.iters if iters is None else iters
        return iters * self.cfg.K * B, (2 * self.cfg.K if iters else self.cfg.K) * B


def _mlp(sizes, out_scale=1.0):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        lin = nn.Linear(a, b)
        nn.init.orthogonal_(lin.weight, gain=2 ** 0.5)
        nn.init.zeros_(lin.bias)
        if i == len(sizes) - 2:
            with torch.no_grad():
                lin.weight.mul_(out_scale)
        layers += [lin] if i == len(sizes) - 2 else [lin, nn.ELU()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """NablaMimic's encoder-decoder actor, the anchor orientation going to the decoder.

    z = tanh(enc(future reference frames)), action = dec([z, proprioception history, anchor orientation]).
    """

    def __init__(self, goal_ref_dim, hist_dim, anchor_dim, nu, hidden=(2048, 1024, 512, 256), latent=32):
        super().__init__()
        self.enc = _mlp([goal_ref_dim, *hidden, latent])
        self.dec = _mlp([latent + hist_dim + anchor_dim, *hidden, nu], out_scale=0.01)

    def forward(self, hist, goal_ref, anchor):
        z = torch.tanh(self.enc(goal_ref))
        return self.dec(torch.cat([z, hist, anchor], -1))
