r"""MENTOR: Mimicking Expert states by iNverting the simulaTOR.

Imitation from state-only demonstrations that were produced by the same differentiable
simulator the learner can query. The simulator plus the demonstrations act as an
interactive expert ("DAgger without an expert"):

1. Invert: recover the action behind every demonstration transition (s, s') by
   Levenberg-Marquardt on min_a ||W (f(s, a) - s')||^2 (finite-difference Jacobian).
2. Rejoin teacher: for any state x the learner visits, plan K open-loop actions in the
   simulator so that the state after K steps lands back on a demonstration (nearest demo
   windows, phase offsets, warm start from the recovered actions, Adam through the sim).
   The first planned action is the label.
3. Distill: DAgger. Behavioural cloning on the recovered actions, then per round: learner
   rollouts, teacher labels, a cost gate, aggregation and supervised regression.

State-only: the demonstrations' stored actions are dropped on load and never read. Demo
rewards are only used to report expert-normalized returns.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from ...common.demos import get_demos
from ..agent import Agent
from .mentor_sim import OneStep, SimCounter, SynergySpec, get_spec


class MentorPolicy(nn.Module):
    """Deterministic MLP policy, input standardised with the demo statistics.

    ``squash='tanh'``: tanh output. ``squash='clip'``: linear output while training (the regression
    keeps its gradient for labels at the action bounds, e.g. muscles that are fully off) and clipped
    to [-1, 1] everywhere else (rollouts, teacher queries, evaluation). ``fitting`` is set by fit().
    """

    def __init__(self, obs_dim, act_dim, mean, std, units=(256, 256), squash='tanh'):
        super().__init__()
        if squash not in ('tanh', 'clip'):
            raise ValueError(f'unknown policy squash {squash!r}')
        self.squash = squash
        self.fitting = False
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clone())
        layers, d = [], obs_dim
        for u in units:
            layers += [nn.Linear(d, u), nn.ELU()]
            d = u
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        out = self.net((obs - self.mean) / self.std)
        if self.squash == 'tanh':
            return torch.tanh(out)
        return out if self.fitting else out.clamp(-1, 1)


@torch.no_grad()
def lm_invert(
    step,
    s,
    s1,
    w,
    iters=5,
    starts=1,
    h=1e-2,
    lam0=1e-2,
    reg=0.0,
    param='box',
    generator=None,
    jacobian='fd',
    refresh=0,
    reg_kind='l2',
):
    """Batched Levenberg-Marquardt for min_a ||w * (step(s, a) - s1)||^2 + reg ||a||^2 with |a| <= 1.

    Forward-difference Jacobian (act_dim + 1 forward steps per iteration). The first start
    is a = 0, later ones are uniform in (-1, 1); the best solution per sample is kept.
    ``param='box'`` clips each step to [-1, 1]; ``param='tanh'`` optimises u with a = tanh(u),
    which avoids the clipping distortion when many actions sit at a bound. ``reg > 0`` picks
    one consistent (minimum-norm) solution when the actions outnumber what the next state
    determines, as for muscle-driven bodies (SNU: 152 muscles, 18 dofs).
    ``jacobian='broyden'`` computes the forward-difference Jacobian only at the first iteration of
    each start (and every ``refresh`` iterations if > 0) and otherwise applies Broyden's rank-1
    update, so an iteration costs one forward step instead of act_dim + 1.
    ``reg_kind='activation'`` replaces reg ||a||^2 (which favours a = 0, i.e. half-activated muscles) by
    reg * sum (a + 1) / 2, the total muscle activation (the effort that muscle-driven experts minimise).
    Returns actions (n, act_dim) and the residual part ||w * (step(s, a) - s1)||^2 (n,).
    """
    if jacobian not in ('fd', 'broyden'):
        raise ValueError(f'unknown inversion Jacobian {jacobian!r}')
    if reg_kind not in ('l2', 'activation'):
        raise ValueError(f'unknown inversion reg_kind {reg_kind!r}')
    act_term = reg_kind == 'activation'

    def penalty(a):
        return ((a + 1) * 0.5).sum(-1) if act_term else (a**2).sum(-1)

    if param not in ('box', 'tanh'):
        raise ValueError(f'unknown inversion parametrisation {param!r}')
    n, act_dim = s.shape[0], step.spec.act_dim
    tanh = param == 'tanh'
    to_action = torch.tanh if tanh else (lambda x: x)

    def residual(x):
        return (step(s, to_action(x), grad=False) - s1) * w

    best_a, best_c, best_total = None, None, None
    for k in range(starts):
        if k == 0:
            x = torch.zeros(n, act_dim, device=s.device)
        else:
            x = torch.rand(n, act_dim, device=s.device, generator=generator) * 2 - 1
            x = torch.atanh(0.9 * x) if tanh else x
        r = residual(x)
        a = to_action(x)
        cost = (r**2).sum(-1)
        total = cost + reg * penalty(a)
        lam = torch.full_like(cost, lam0)
        J = None
        for it in range(iters):
            if J is None or jacobian == 'fd' or (refresh > 0 and it % refresh == 0):
                cols = []
                for j in range(act_dim):
                    xp = x.clone()
                    if tanh:
                        xp[:, j] = x[:, j] + h
                    else:
                        sign = torch.where(x[:, j] + h > 1, -1.0, 1.0)
                        xp[:, j] = (x[:, j] + sign * h).clamp(-1, 1)
                    dx = (xp[:, j] - x[:, j]).unsqueeze(-1)
                    cols.append((residual(xp) - r) / dx)
                J = torch.stack(cols, -1)
            a = to_action(x)
            da = 1 - a**2 if tanh else torch.ones_like(a)  # d a / d x
            if act_term:  # linear penalty: gradient only (LM damping regularises the step)
                H = J.transpose(1, 2) @ J
                g = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1) + 0.25 * reg * da
            else:
                H = J.transpose(1, 2) @ J + reg * torch.diag_embed(da**2)
                g = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1) + reg * a * da
            A = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-6 if tanh else 1e-12))
            x_new = x - torch.linalg.solve(A, g.unsqueeze(-1)).squeeze(-1)
            x_new = x_new if tanh else x_new.clamp(-1, 1)
            r_new = residual(x_new)
            c_new = (r_new**2).sum(-1)
            t_new = c_new + reg * penalty(to_action(x_new))
            ok = t_new < total
            if jacobian == 'broyden':  # secant update from this trial step, accepted or not
                dx = x_new - x
                denom = (dx**2).sum(-1, keepdim=True).clamp_min(1e-12)
                J = J + ((r_new - r - (J @ dx.unsqueeze(-1)).squeeze(-1)) / denom).unsqueeze(-1) * dx.unsqueeze(1)
            x = torch.where(ok[:, None], x_new, x)
            r = torch.where(ok[:, None], r_new, r)
            cost = torch.where(ok, c_new, cost)
            total = torch.where(ok, t_new, total)
            lam = torch.where(ok, (lam * 0.3).clamp_min(1e-9), (lam * 5.0).clamp_max(1e9))
        a = to_action(x)
        if best_a is None:
            best_a, best_c, best_total = a, cost, total
        else:
            ok = total < best_total
            best_a = torch.where(ok[:, None], a, best_a)
            best_c = torch.where(ok, cost, best_c)
            best_total = torch.where(ok, total, best_total)
    return best_a, best_c


class RejoinTeacher:
    """K-step open-loop plan from a learner state back onto the demonstration manifold.

    cost(a) = min_tau ||w (x_K(a) - s'_{j+K-1+tau})||^2 + lam ||a - a_ref||^2, where (traj, j)
    is a nearest demo window, tau ranges over ``phase_offsets`` and a_ref are the recovered
    demo actions of that window. ``screen``: rank the ``num_windows`` nearest windows by the
    warm-start cost (forward only) and optimise only the best one, otherwise optimise all
    and keep the best.
    """

    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        self.spec, self.device = spec, device
        self.S, self.S1, self.A_hat, self.lengths = S, S1, A_hat, lengths
        self.batch = batch
        self.K = int(cfg.horizon)
        self.m = min(int(cfg.num_windows), S.shape[0])  # one window per demo trajectory
        self.iters = int(cfg.iters)
        self.lr = float(cfg.lr)
        self.lam = float(cfg.action_reg)
        self.screen = bool(cfg.screen) and self.m > 1
        # Policy anchor (optional): warm-start and regularise the plan towards the current policy's
        # closed-loop actions (mixed with the demo actions by anchor_mix), so that over-actuated
        # bodies get labels consistent with what the policy already does. Set policy via anchor_to().
        self.anchor_mix = float(cfg.get('anchor_mix', 0.0))
        self.anchor_policy = None
        # Optional terms on the planned actions: effort (activation: sum of (a + 1) / 2, i.e. the muscle activations
        # of the SNU convention; l2: sum of a^2) and temporal smoothness sum_k ||u_k - u_{k-1}||^2 within the plan.
        self.effort = float(cfg.get('effort_weight', 0.0))
        self.effort_kind = str(cfg.get('effort_kind', 'activation'))
        if self.effort_kind not in ('activation', 'l2'):
            raise ValueError(f'unknown teacher effort_kind {self.effort_kind!r}')
        self.smooth = float(cfg.get('smooth_weight', 0.0))
        self.offsets = torch.tensor(list(cfg.phase_offsets), device=device)
        N, T, D = S.shape
        self.N, self.T, self.D = N, T, D
        valid = torch.arange(T, device=device)[None] < lengths[:, None]
        self.w = 1.0 / S[valid].std(0).clamp_min(1e-6)
        # Optional: observation coordinates the teacher ignores when matching windows and in its cost, e.g. a goal
        # pose the actions cannot change (AllegroHand: "45-52"). Comma-separated 'a-b' (or 'a:b') ranges or indices;
        # prefer 'a-b' on command lines: YAML 1.1 reads an unquoted 45:52 as the base-60 number 2752.
        for part in [p for p in str(cfg.get('ignore_dims', '') or '').split(',') if p.strip()]:
            a_, _, b_ = part.strip().replace(':', '-').partition('-')
            self.w[int(a_) : (int(b_) if b_ else int(a_) + 1)] = 0.0
        self.S_flat = (S * self.w).reshape(-1, D)
        # Support barrier (optional): penalise planned states outside the per-coordinate range
        # the demos visit (widened by support_margin standard deviations), e.g. a torso too low.
        self.barrier = float(cfg.get('support_barrier', 0.0))
        spread = float(cfg.get('support_margin', 0.5)) / self.w
        self.lo, self.hi = S[valid].min(0).values - spread, S[valid].max(0).values + spread
        # admissible window starts: a full K-step window plus margin for the phase offsets
        t = torch.arange(T, device=device)[None]
        margin = int(self.offsets.max().clamp_min(0).item()) + 2
        self.window_ok = (t >= 3) & (t <= (lengths[:, None] - self.K - margin))
        if not self.window_ok.any():
            raise ValueError(f'demonstrations too short for a rejoin horizon of {self.K}')
        n_plan = batch * self.m
        self.sim_all = self.make_sim(spec, n_plan, device, counter)
        self.sim_one = self.make_sim(spec, batch, device, counter) if self.screen else None

    def make_sim(self, spec, n, device, counter):
        return OneStep(spec, n, device, counter)

    def extend(self, S_new, S1_new, A_new, max_extra):
        """Append short trajectories [k, L, ...] to the library, padded to its length T.

        Keeps the original demos and at most max_extra added trajectories (oldest dropped first).
        """
        if not hasattr(self, 'n_original'):
            self.n_original = self.N
        k, L = S_new.shape[:2]
        pad = self.T - L
        if pad < 0:
            S_new, S1_new, A_new, L, pad = S_new[:, : self.T], S1_new[:, : self.T], A_new[:, : self.T], self.T, 0
        padf = lambda x: torch.cat([x, x.new_zeros(k, pad, *x.shape[2:])], 1)
        self.S = torch.cat([self.S, padf(S_new)])
        self.S1 = torch.cat([self.S1, padf(S1_new)])
        self.A_hat = torch.cat([self.A_hat, padf(A_new)])
        self.lengths = torch.cat([self.lengths, torch.full((k,), L, device=self.device, dtype=self.lengths.dtype)])
        extra = self.S.shape[0] - self.n_original
        if extra > max_extra:
            keep = torch.cat(
                [
                    torch.arange(self.n_original, device=self.device),
                    torch.arange(self.S.shape[0] - max_extra, self.S.shape[0], device=self.device),
                ]
            )
            self.S, self.S1, self.A_hat, self.lengths = self.S[keep], self.S1[keep], self.A_hat[keep], self.lengths[keep]
        self.N = self.S.shape[0]
        self.S_flat = (self.S * self.w).reshape(-1, self.D)
        t = torch.arange(self.T, device=self.device)[None]
        margin = int(self.offsets.max().clamp_min(0).item()) + 2
        self.window_ok = (t >= 3) & (t <= (self.lengths[:, None] - self.K - margin))

    def windows(self, x):
        d = torch.cdist(x * self.w, self.S_flat).view(-1, self.N, self.T)
        d = d.masked_fill(~self.window_ok[None], float('inf'))
        dmin, jmin = d.min(-1)
        top = dmin.topk(self.m, dim=1, largest=False).indices
        return top, jmin.gather(1, top)

    def a_ref(self, traj, j):
        return torch.stack([self.A_hat[traj, j + k] for k in range(self.K)], 1)

    def anchor_to(self, policy):
        self.anchor_policy = policy if self.anchor_mix > 0 else None

    @torch.no_grad()
    def anchored(self, sim, x, a_demo):
        """Reference actions: the demo window's, mixed with the policy's K closed-loop actions from x."""
        if self.anchor_policy is None:
            return a_demo
        y, acts = x, []
        for k in range(self.K):
            a = self.anchor_policy(y).clamp(-1, 1)
            acts.append(a)
            if k < self.K - 1:
                y = sim(y, a, grad=False)
        return (1 - self.anchor_mix) * a_demo + self.anchor_mix * torch.stack(acts, 1)

    def rollout(self, sim, x, a, grad):
        """Final state and the support-barrier penalty accumulated along the plan."""
        y, outside = x, 0.0
        for k in range(self.K):
            y = sim(y, a[:, k], grad=grad)
            if self.barrier > 0:
                outside = outside + (((self.lo - y).clamp_min(0) + (y - self.hi).clamp_min(0)) * self.w).pow(2).sum(-1)
        return y, outside

    def cost(self, rolled, a, traj, j, a_ref):
        y, outside = rolled
        idx = j[:, None] + self.K - 1 + self.offsets[None]
        idx = torch.minimum(idx.clamp_min(0), (self.lengths[traj] - 1)[:, None])
        target = self.S1[traj[:, None], idx]
        err = (((y[:, None] - target) * self.w) ** 2).sum(-1).min(-1).values
        c = err + self.lam * ((a - a_ref) ** 2).sum((-1, -2)) + self.barrier * outside
        if self.effort > 0:
            e = (a.clamp(-1, 1) + 1) * 0.5 if self.effort_kind == 'activation' else a**2
            c = c + self.effort * e.sum((-1, -2))
        if self.smooth > 0 and a.shape[1] > 1:
            c = c + self.smooth * ((a[:, 1:] - a[:, :-1]) ** 2).sum((-1, -2))
        return c

    def optimise(self, sim, x, traj, j, a_ref):
        a = a_ref.clone().requires_grad_(True)
        opt = torch.optim.Adam([a], lr=self.lr)
        best = torch.full((a.shape[0],), float('inf'), device=self.device)
        best_a = a_ref.clone()
        for it in range(self.iters + 1):
            last = it == self.iters
            with torch.set_grad_enabled(not last):
                c = self.cost(self.rollout(sim, x, a, grad=not last), a, traj, j, a_ref)
            with torch.no_grad():
                better = c < best
                best = torch.where(better, c.detach(), best)
                best_a[better] = a.detach()[better].clamp(-1, 1)
            if last:
                break
            opt.zero_grad()
            c.sum().backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(-1, 1)
        return best_a, best

    def label(self, x):
        """Labels (batch, act_dim) and their rejoin costs (batch,) for learner states x."""
        x = x.detach()
        n, m = x.shape[0], self.m
        top, j = self.windows(x)
        ar = torch.arange(n, device=self.device)
        xr = x[:, None].expand(n, m, self.D).reshape(-1, self.D)
        traj_all, j_all = top.reshape(-1), j.reshape(-1)
        a_ref_all = self.a_ref(traj_all, j_all)
        if self.screen:
            with torch.no_grad():
                c0 = self.cost(self.rollout(self.sim_all, xr, a_ref_all, grad=False), a_ref_all, traj_all, j_all, a_ref_all)
            sel = c0.view(n, m).argmin(1)
            traj, jj = top[ar, sel], j[ar, sel]
            a_ref = self.a_ref(traj, jj)
            if self.anchor_policy is not None:
                a_ref = self.anchored(self.sim_one, x, a_ref)
            if self.iters == 0:
                return a_ref[:, 0], c0.view(n, m)[ar, sel]
            best_a, best = self.optimise(self.sim_one, x, traj, jj, a_ref)
            self.last_choice = (x, best_a, traj, jj)
            return best_a[:, 0], best
        if self.anchor_policy is not None:
            a_ref_all = self.anchored(self.sim_all, xr, a_ref_all)
        best_a, best = self.optimise(self.sim_all, xr, traj_all, j_all, a_ref_all)
        sel = best.view(n, m).argmin(1)
        plan = best_a.view(n, m, self.K, -1)[ar, sel]
        self.last_choice = (x, plan, top[ar, sel], j[ar, sel])
        return plan[:, 0], best.view(n, m)[ar, sel]

    def terminal_error(self, y, traj, j):
        idx = j[:, None] + self.K - 1 + self.offsets[None]
        idx = torch.minimum(idx.clamp_min(0), (self.lengths[traj] - 1)[:, None])
        return (((y[:, None] - self.S1[traj[:, None], idx]) * self.w) ** 2).sum(-1).min(-1).values

    @torch.no_grad()
    def advantage(self, policy):
        """Rejoin errors of the teacher's plan and of the policy after K steps.

        From the states and windows of the last label() call; the policy runs closed loop. The
        teacher helps where its error is lower.
        """
        x, plan, traj, j = self.last_choice
        if getattr(self, 'sim_eval', None) is None:
            self.sim_eval = (
                self.sim_one if self.sim_one is not None else OneStep(self.spec, self.batch, self.device, self.sim_all.counter)
            )
        y_t, y_p = x, x
        for k in range(self.K):
            y_t = self.sim_eval(y_t, plan[:, k], grad=False)
            y_p = self.sim_eval(y_p, policy(y_p), grad=False)
        return self.terminal_error(y_t, traj, j), self.terminal_error(y_p, traj, j)


class MENTOR(Agent):
    r"""MENTOR: Mimicking Expert states by iNverting the simulaTOR (see module docstring)."""

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.mentor_config = cfg = full_cfg.agent.mentor
        self.num_actors = int(cfg.num_actors)
        super().__init__(full_cfg, **kwargs)
        if self.normalize_input:
            raise ValueError('MENTOR standardises inputs with the demo statistics; set network.normalize_input=False')

        self.seed = int(full_cfg.seed)
        self.env_name = full_cfg.task.env.env_name
        if str(full_cfg.task.name) == 'Rewarped':  # full-state mode: plans start from saved simulator states
            from .mentor_rewarped import RewarpedSpec

            self.spec = RewarpedSpec(
                full_cfg, cfg.get('obs_keys', 'joint_q|com_q'), str(cfg.get('stage1', 'joint_velocity')), self.device
            )
        else:
            self.spec = get_spec(self.env_name)
        self.full_state = bool(getattr(self.spec, 'full_state', False))
        self.state_bank = None
        self.counter = SimCounter()  # Stage I + teacher simulator transitions
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        # --- Demonstrations (state only) ---
        demos = get_demos(self.device, **cfg.demos)
        stored_actions = demos.pop('act')  # never read (state-only imitation), except by the flagged audit below
        self.expert_return = float(demos['expert_return'])
        D = self.spec.obs_dim
        obs, next_obs = demos['obs'], demos['next_obs']
        if self.full_state:  # selected observation keys, flattened per step
            N_, T_ = demos['done'].shape[:2]
            ks = self.spec.key_slices
            self.S = torch.cat([obs[k].reshape(N_, T_, -1)[..., ks[k]] for k in self.spec.obs_keys], -1).float().contiguous()
            self.S1 = torch.cat([next_obs[k].reshape(N_, T_, -1)[..., ks[k]] for k in self.spec.obs_keys], -1)
            self.S1 = self.S1.float().contiguous()
        else:
            obs = obs['obs'] if isinstance(obs, dict) else obs
            next_obs = next_obs['obs'] if isinstance(next_obs, dict) else next_obs
            self.S = obs[..., :D].float().contiguous()
            self.S1 = next_obs[..., :D].float().contiguous()
        self.lengths = self._lengths(demos['done'])
        # End each demo at its first non-finite step (e.g. the SNU simulator goes NaN at the terminal step of a fall);
        # Rewarped zeroes the observations of envs that went NaN instead. One NaN target would otherwise reach the
        # residual weights below and every Stage I residual.
        bad = ~torch.isfinite(self.S).all(-1) | ~torch.isfinite(self.S1).all(-1)
        if self.full_state:
            bad |= (self.S.abs().sum(-1) == 0) | (self.S1.abs().sum(-1) == 0)
        valid_before = torch.arange(bad.shape[1], device=bad.device)[None] < self.lengths[:, None]
        n_cut = 0
        for i in range(bad.shape[0]):
            idx = (bad[i] & valid_before[i]).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                self.lengths[i] = min(int(self.lengths[i]), int(idx[0]))
                n_cut += 1
        if n_cut:
            print(f'[MENTOR] {n_cut} demonstrations cut at their first non-finite or zeroed step')
        self.S, self.S1 = torch.nan_to_num(self.S), torch.nan_to_num(self.S1)  # beyond the cut, never used
        # AUDIT ONLY (invert.audit_true_actions): replace the recovered actions by the stored expert actions, to
        # measure what action recovery costs. This reads demonstrator actions: never use it for reported results.
        # True / 'both': the Stage I labels and the teacher's reference actions; 'labels' or 'teacher': only one of them.
        self.audit_actions, self.audit_mode = None, None
        mode = cfg.invert.get('audit_true_actions', False)
        mode = 'both' if mode is True else (None if mode in (False, None, 'none') else str(mode))
        if mode is not None:
            if mode not in ('both', 'labels', 'teacher'):
                raise ValueError(f'unknown invert.audit_true_actions {mode!r}')
            self.audit_actions, self.audit_mode = stored_actions[..., : self.spec.act_dim].float().clamp(-1, 1).contiguous(), mode
            print(f'[MENTOR] AUDIT ({mode}): using the STORED EXPERT ACTIONS (diagnostic run, not state-only)')
        del stored_actions
        T = self.S.shape[1]
        self.valid = torch.arange(T, device=self.device)[None] < self.lengths[:, None]
        S_valid = self.S[self.valid]
        self.w_step = 1.0 / (self.S1[self.valid] - S_valid).std(0).clamp_min(1e-8)

        # --- Policy ---
        units = tuple(self.network_config.get('units', [256, 256]))
        n_syn = int(cfg.get('synergies', {}).get('dims', 0))
        self.policy = MentorPolicy(
            D,
            n_syn or self.spec.act_dim,
            S_valid.mean(0),
            S_valid.std(0).clamp_min(1e-6),
            units,
            squash=str(self.network_config.get('squash', 'tanh')),
        )
        self.policy = self.policy.to(self.device)
        print(self.policy, '\n')

        # --- State ---
        self.A_hat = None
        self.data_s = None
        self.data_a = None
        self.teacher = None
        self.round = 0
        self.stage1_steps = 0
        self.t_start = None
        self.wall_offset = 0.0
        self.obs = None

    # ---- helpers ----
    def _lengths(self, done):
        lengths = torch.full((done.shape[0],), done.shape[1], device=self.device, dtype=torch.long)
        for i in range(done.shape[0]):
            idx = (done[i] == 1).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                lengths[i] = idx[0] + 1
        return lengths

    def _state_obs(self, obs):
        if self.full_state:
            return self.spec.obs_vec(obs)
        obs = obs['obs'] if isinstance(obs, dict) else obs
        return obs[:, : self.spec.obs_dim]

    def _wall(self):
        return self.wall_offset + (time.time() - self.t_start if self.t_start is not None else 0.0)

    def _sim_stats(self):
        learner = int(self.agent_steps)
        return {
            'sim_steps/learner': learner,
            'sim_steps/stage1': int(self.stage1_steps),
            'sim_steps/teacher_fwd': int(self.counter.fwd - self.stage1_steps),
            'sim_steps/teacher_fwdbwd': int(self.counter.fwdbwd),
            'sim_steps/total': learner + int(self.counter.total),
        }

    def env_action(self, a):
        """Policy/teacher action → the env's action (decodes synergies when they are on)."""
        return self.spec.decode(a) if isinstance(self.spec, SynergySpec) else a

    def use_synergies(self, mean, basis, scale):
        self.spec_base = getattr(self, 'spec_base', self.spec)
        self.spec = SynergySpec(self.spec_base, mean, basis, scale)

    def fit_synergies(self):
        """PCA of the recovered actions; switch the teacher, labels and policy to synergy space."""
        d = int(self.mentor_config.synergies.dims)
        A = self.A_hat[self.valid]
        mean = A.mean(0)
        U, S, V = torch.linalg.svd(A - mean, full_matrices=False)
        basis = V[:d].T  # (act_dim, d)
        scale = 3.0 * (S[:d] / (A.shape[0] - 1) ** 0.5)  # 3 std per component
        explained = float((S[:d] ** 2).sum() / (S**2).sum())
        self.use_synergies(mean, basis, scale)
        z = torch.zeros(*self.A_hat.shape[:2], d, device=self.device)
        z[self.valid] = self.spec.encode(A)
        self.A_hat = self.A_labels = z
        print(f'[MENTOR] synergies: {d} of {A.shape[1]} dims, explained variance {explained:.3f}')
        return {'synergies/explained_variance': explained}

    def get_actions(self, obs, sample: bool = False):
        return self.env_action(self.policy(self._state_obs(obs)))

    # ---- Stage I ----
    def invert(self):
        cfg = self.mentor_config.invert
        batch = int(cfg.batch)
        s, s1 = self.S[self.valid], self.S1[self.valid]
        n = s.shape[0]
        if self.full_state and self.spec.stage1 != 'invert':  # actions read off the observations, no simulation
            t0 = time.time()
            A_hat = torch.zeros(*self.S.shape[:2], self.spec.act_dim, device=self.device)
            A_hat[self.valid] = self.spec.recover_actions(s, s1)
            self.A_hat = self.A_labels = A_hat
            self.stage1_steps = 0
            print(f'[MENTOR] Stage I ({self.spec.stage1}): {n} transitions, no simulation')
            return {'stage1/sim_steps': 0, 'stage1/time': time.time() - t0}
        if self.full_state:  # Levenberg-Marquardt through Rewarped via the adapter's observation-to-state map
            step = self.spec.make_obs_step(batch, self.device, self.counter)
        else:
            step = OneStep(self.spec, batch, self.device, self.counter)
        acts, costs = [], []
        t0 = time.time()
        before = self.counter.total
        for c0 in range(0, n, batch):
            x, y = s[c0 : c0 + batch], s1[c0 : c0 + batch]
            k = x.shape[0]
            if k < batch:  # OneStep has a fixed batch: pad with copies of the first row
                x = torch.cat([x, x[:1].expand(batch - k, -1)])
                y = torch.cat([y, y[:1].expand(batch - k, -1)])
            a, c = lm_invert(
                step,
                x,
                y,
                self.w_step,
                iters=int(cfg.iters),
                starts=int(cfg.starts),
                reg=float(cfg.get('reg', 0.0)),
                param=str(cfg.get('param', 'box')),
                generator=self.generator,
                jacobian=str(cfg.get('jacobian', 'fd')),
                refresh=int(cfg.get('refresh', 0)),
                reg_kind=str(cfg.get('reg_kind', 'l2')),
            )
            acts.append(a[:k])
            costs.append(c[:k])
        del step
        torch.cuda.empty_cache()
        self.stage1_steps = self.counter.total - before
        a_flat, cost = torch.cat(acts), torch.cat(costs)
        A_hat = torch.zeros(*self.S.shape[:2], self.spec.act_dim, device=self.device)
        A_hat[self.valid] = a_flat
        self.A_hat = self.A_labels = A_hat  # teacher reference actions and Stage I labels
        if self.audit_actions is not None:
            A_true = torch.zeros_like(A_hat)
            A_true[self.valid] = self.audit_actions[self.valid]
            print(f'[MENTOR] AUDIT: rms(recovered - stored) = {((a_flat - A_true[self.valid]) ** 2).mean().sqrt().item():.3f}')
            if self.audit_mode in ('both', 'teacher'):
                self.A_hat = A_true
            if self.audit_mode in ('both', 'labels'):
                self.A_labels = A_true
        rel = cost.sqrt() / ((s1 - s) * self.w_step).norm(dim=-1).clamp_min(1e-9)
        q = torch.quantile(rel, torch.tensor([0.5, 0.95, 0.99], device=self.device))
        stats = {
            'stage1/rel_residual_q50': q[0].item(),
            'stage1/rel_residual_q95': q[1].item(),
            'stage1/rel_residual_q99': q[2].item(),
            'stage1/frac_rel_residual_gt_1e-2': (rel > 1e-2).float().mean().item(),
            'stage1/sim_steps': int(self.stage1_steps),
            'stage1/time': time.time() - t0,
        }
        print(
            f'[MENTOR] Stage I: {n} transitions, {self.stage1_steps} sim steps, {stats["stage1/time"]:.0f}s, rel residual q50/95/99 {q.tolist()}'
        )
        return stats

    # ---- Stage III ----
    @torch.no_grad()
    def sync_jacobians(self, h=1e-2, batch=1000):
        """Weighted action Jacobians W df/da at every dataset (state, label) pair not covered yet.

        Forward differences through one simulator step (act_dim + 1 forward steps per sample),
        with W = 1/std(s' - s) as in Stage I. Used by the effect-space fit loss.
        """
        n_have = 0 if getattr(self, 'data_J', None) is None else self.data_J.shape[0]
        n_all = self.data_s.shape[0]
        if n_have >= n_all:
            return
        if getattr(self, 'jac_step', None) is None:
            self.jac_step = OneStep(self.spec, batch, self.device, self.counter)
        out = []
        for c0 in range(n_have, n_all, batch):
            s = self.data_s[c0 : c0 + batch]
            a = self.data_a[c0 : c0 + batch].clamp(-1, 1)
            k = s.shape[0]
            if k < batch:
                s = torch.cat([s, s[:1].expand(batch - k, -1)])
                a = torch.cat([a, a[:1].expand(batch - k, -1)])
            y = self.jac_step(s, a, grad=False)
            cols = []
            for j in range(self.spec.act_dim):
                ap = a.clone()
                sign = torch.where(a[:, j] + h > 1, -1.0, 1.0)
                ap[:, j] = (a[:, j] + sign * h).clamp(-1, 1)
                da = (ap[:, j] - a[:, j]).unsqueeze(-1)
                cols.append((self.jac_step(s, ap, grad=False) - y) * self.w_step / da)
            out.append(torch.stack(cols, -1)[:k])
        new = torch.cat(out)
        self.data_J = new if n_have == 0 else torch.cat([self.data_J, new])

    def fit(self, steps, lr, round_idx):
        cfg = self.mentor_config.fit
        effect = str(cfg.get('loss', 'action')) == 'effect'
        if effect:  # fit the action's effect on the next state, ||W J (pi(s) - a*)||^2 (+ small action term)
            self.sync_jacobians()
            action_weight = float(cfg.get('action_weight', 1e-3))
        opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=float(cfg.lr_min))
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed * 100 + round_idx)
        bs = int(cfg.batch_size)
        self.policy.train()
        self.policy.fitting = True
        for _ in range(steps):
            idx = torch.randint(0, self.data_s.shape[0], (bs,), device=self.device, generator=g)
            diff = self.policy(self.data_s[idx]) - self.data_a[idx]
            if effect:
                moved = (self.data_J[idx] @ diff.unsqueeze(-1)).squeeze(-1)
                loss = (moved**2).sum(-1).mean() + action_weight * (diff**2).sum(-1).mean()
            else:
                loss = (diff**2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
        self.policy.fitting = False
        self.policy.eval()
        with torch.no_grad():
            full = (
                sum(
                    ((self.policy(self.data_s[i : i + 65536]) - self.data_a[i : i + 65536]) ** 2).sum()
                    for i in range(0, self.data_s.shape[0], 65536)
                )
                / self.data_s.shape[0]
            )
        return float(full)

    @torch.no_grad()
    def collect(self):
        cfg = self.mentor_config.rollout
        T, noise, reset_every = int(cfg.length), float(cfg.noise), int(cfg.reset_every)
        frac = float(cfg.get('reset_fraction', 1.0))
        if self.obs is None:
            self.obs = self._state_obs(self._convert_obs(self.env.reset()))
        n_reset = int(round(frac * self.num_actors))
        visited, nexts, acts, breaks = [], [], [], []
        if self.full_state:  # save the full state of the (step, env) pairs the teacher will label
            tcfg = self.mentor_config.teacher
            need = int(tcfg.batch) * int(tcfg.batches_per_round)
            if need > T * self.num_actors:
                raise ValueError(f'rollouts give {T * self.num_actors} states, the teacher needs {need}')
            pick = torch.randperm(T * self.num_actors, device=self.device, generator=self.generator)[:need]
            bank_obs, bank_states = [], []
        for t in range(T):
            broken = torch.zeros(self.num_actors, dtype=torch.bool, device=self.device)
            if reset_every > 0 and t % reset_every == 0 and n_reset > 0:
                ids = torch.arange(n_reset, device=self.device)
                self.env.reset(env_ids=ids)
                ob = self.env.obs_buf
                ob = {k: v.clone() for k, v in ob.items()} if isinstance(ob, dict) else ob.clone()
                self.obs = self._state_obs(self._convert_obs(ob))
                broken[ids] = t > 0
            a = self.policy(self.obs)
            if noise > 0:
                a = (a + noise * torch.randn(a.shape, device=self.device, generator=self.generator)).clamp(-1, 1)
            visited.append(self.obs.clone())
            if self.full_state:
                ids = pick[pick // self.num_actors == t] % self.num_actors
                if len(ids) > 0:
                    bank_obs.append(self.obs[ids].detach().clone())
                    bank_states.append(self.spec.get_states(self.env, ids))
            obs, rew, done, info = self.env.step(self.env_action(a))
            before_reset = info.get('obs_before_reset') if isinstance(info, dict) else None
            before_reset = obs if before_reset is None else before_reset  # Rewarped sets None when no env reset
            nexts.append(self._state_obs(self._convert_obs(before_reset)).clone())
            acts.append(a.clone())
            breaks.append(broken | done.bool())
            self.obs = self._state_obs(self._convert_obs(obs)).clone()
            self.agent_steps += self.num_actors
        # per-step tensors [T, n, ...] for library bootstrapping
        self.last_rollout = {
            'obs': torch.stack(visited),
            'next': torch.stack(nexts),
            'act': torch.stack(acts),
            'broken': torch.stack(breaks),
        }
        if self.full_state:
            self.state_bank = (torch.cat(bank_obs), self.spec.cat_states(bank_states))
        return torch.cat(visited)

    def bootstrap_library(self, cost):
        """Add learner segments that stayed up and close to the demos to the teacher's library.

        A segment is one env's rollout of this round (bootstrap.min_len steps or more, no fall or
        reset); it is accepted when every labelled state in it has rejoin cost <= bootstrap.tau.
        Its states, next states and executed actions become a new (short) library trajectory.
        """
        bcfg = self.mentor_config.get('bootstrap', {})
        if not bcfg or not bool(bcfg.get('enabled', False)):
            return {}
        roll, n = self.last_rollout, self.num_actors
        T = roll['obs'].shape[0]
        idx = self.last_label_index
        env_of = idx % n
        q = float(bcfg.get('quantile', 0.8))  # robust per-segment cost (1.0 = the worst labelled state)
        seg_cost = torch.full((n,), float('inf'), device=self.device)
        labelled = torch.zeros(n, device=self.device).scatter_add(0, env_of, torch.ones_like(cost))
        for e in range(n):
            c = cost[env_of == e]
            if len(c):
                seg_cost[e] = c.quantile(q)
        intact = ~roll['broken'].any(0)
        ok = intact & (labelled >= int(bcfg.get('min_labelled', 8))) & (seg_cost <= float(bcfg.get('tau', 0.1)))
        envs = ok.nonzero().flatten()[: int(bcfg.get('max_new', 8))]
        if len(envs) and T >= self.teacher.K + 12:
            self.teacher.extend(
                roll['obs'][:, envs].transpose(0, 1),
                roll['next'][:, envs].transpose(0, 1),
                roll['act'][:, envs].transpose(0, 1),
                int(bcfg.get('max_extra', 64)),
            )
        finite = seg_cost[intact & torch.isfinite(seg_cost)]
        return {
            'train/library_trajs': self.teacher.N,
            'train/library_added': int(len(envs)),
            'train/segments_ok': int(ok.sum()),
            'train/segments_intact': int(intact.sum()),
            'train/segment_cost_median': float(finite.median()) if len(finite) else float('nan'),
        }

    def teacher_demonstrations(self):
        """Roll out the closed-loop teacher from random initial states before DAgger.

        Trajectories that never fall and whose rejoin-cost quantile is <= bootstrap.teacher_tau
        join the teacher's library and their (state, teacher action) pairs join the dataset:
        a few demonstrations become many simulator-checked reference trajectories.
        """
        cfg = self.mentor_config
        bcfg = cfg.bootstrap
        steps, n = int(bcfg.teacher_steps), self.num_actors
        tau = float(bcfg.get('teacher_tau', bcfg.get('tau', 0.1)))
        q = float(bcfg.get('quantile', 0.8))
        t0 = time.time()
        driver = RejoinTeacher(self.spec, self.S, self.S1, self.A_hat, self.lengths, n, self.device, self.counter, cfg.teacher)
        env_cfg = self.full_cfg.task.env
        env = self.spec.make_env(
            n,
            self.device,
            seed=self.seed + 777,
            stochastic_init=bool(env_cfg.get('stochastic_init', True)),
            early_termination=bool(env_cfg.get('early_termination', True)),
        )
        with torch.no_grad():
            obs = self._state_obs(env.reset()).detach().clone()
        alive = torch.ones(n, dtype=torch.bool, device=self.device)
        S, S1, A, C = [], [], [], []
        for _ in range(steps):
            a, c = driver.label(obs)
            with torch.no_grad():
                o, _, done, info = env.step(self.env_action(a))
            before = info.get('obs_before_reset', o) if isinstance(info, dict) else o
            S.append(obs)
            S1.append(self._state_obs(before).detach().clone())
            A.append(a.detach())
            C.append(c.detach())
            alive &= ~done.bool()
            obs = self._state_obs(o).detach().clone()
        S, S1, A, C = torch.stack(S, 1), torch.stack(S1, 1), torch.stack(A, 1), torch.stack(C, 1)
        ok = alive & (C.quantile(q, dim=1) <= tau)
        keep = ok.nonzero().flatten()
        if len(keep):
            self.teacher.extend(S[keep], S1[keep], A[keep], int(bcfg.get('max_extra', 64)))
            good = (
                C[keep] <= float(cfg.gate.tau) if float(cfg.gate.quantile) == 0 else torch.ones_like(C[keep], dtype=torch.bool)
            )
            self.data_s = torch.cat([self.data_s, S[keep][good]])
            self.data_a = torch.cat([self.data_a, A[keep][good]])
        del driver, env
        torch.cuda.empty_cache()
        print(
            f'[MENTOR] teacher demonstrations: {int(alive.sum())}/{n} stayed up for {steps} steps, {len(keep)} accepted, '
            f'library {self.teacher.N} trajectories, {time.time() - t0:.0f}s'
        )
        self.writer.add(
            self.agent_steps, {'bootstrap/teacher_alive': int(alive.sum()), 'bootstrap/teacher_accepted': len(keep)}
        )
        self.writer.write()

    @torch.no_grad()
    def replay_demo_states(self, count, steps, noise):
        """Physically consistent states near the demos.

        From a random demo state, replay its recovered actions plus Gaussian noise for 1..steps
        simulator steps (count must be a multiple of teacher.batch).
        """
        B = int(self.mentor_config.teacher.batch)
        if getattr(self, 'replay_step', None) is None:
            self.replay_step = OneStep(self.spec, B, self.device, self.counter)
        out = []
        for _ in range(0, count, B):
            traj = torch.randint(0, self.S.shape[0], (B,), device=self.device, generator=self.generator)
            room = (self.lengths[traj] - steps - 1).clamp_min(1)
            t = (torch.rand(B, device=self.device, generator=self.generator) * room).long()
            k = torch.randint(1, steps + 1, (B,), device=self.device, generator=self.generator)
            x, states = self.S[traj, t], []
            for j in range(steps):
                a = self.A_hat[traj, t + j] + noise * torch.randn(
                    B, self.spec.act_dim, device=self.device, generator=self.generator
                )
                x = self.replay_step(x, a.clamp(-1, 1), grad=False)
                states.append(x)
            out.append(torch.stack(states, 1)[torch.arange(B, device=self.device), k - 1])
        return torch.cat(out)[:count]

    def perturbed_demo_states(self, count):
        """Demo states plus Gaussian noise of augment.perturb_scale demo standard deviations per coordinate."""
        scale = float(self.mentor_config.get('augment', {}).get('perturb_scale', 0.1))
        demo = self.S[self.valid]
        idx = torch.randint(0, demo.shape[0], (count,), device=self.device, generator=self.generator)
        noise = torch.randn(count, demo.shape[1], device=self.device, generator=self.generator)
        return demo[idx] + scale * demo.std(0) * noise

    def label(self, visited, n_batches=None):
        cfg = self.mentor_config.teacher
        batch = int(cfg.batch)
        n_batches = int(cfg.batches_per_round) if n_batches is None else n_batches
        need = batch * n_batches
        if visited.shape[0] < need:
            raise ValueError(f'rollouts give {visited.shape[0]} states, the teacher needs {need}; raise rollout.length')
        perm = torch.randperm(visited.shape[0], device=self.device, generator=self.generator)[:need]
        if self.full_state:  # label the states saved during the rollouts (bank holds exactly `need` of them)
            if self.state_bank is None or self.state_bank[0].shape[0] < need:
                raise ValueError('full-state MENTOR labels the states saved by collect(); the bank is too small')
            perm = torch.arange(need, device=self.device)
            visited = self.state_bank[0]
        self.last_label_index = perm
        xs, labels, costs, adv = [], [], [], []
        use_adv = float(self.mentor_config.gate.get('advantage', 0.0)) > 0
        for c0 in range(0, need, batch):
            x = visited[perm[c0 : c0 + batch]]
            if self.full_state:
                from .mentor_rewarped import RewarpedSpec

                a0, c = self.teacher.label((x, RewarpedSpec.take(self.state_bank[1], perm[c0 : c0 + batch])))
            else:
                a0, c = self.teacher.label(x)
            xs.append(x)
            labels.append(a0.detach())
            costs.append(c.detach())
            if use_adv:
                adv.append(torch.stack(self.teacher.advantage(self.policy), 1))
        if use_adv:  # (teacher error, policy error) per label, consumed by the gate in train()
            self.pending_advantage.append(torch.cat(adv))
        return torch.cat(xs), torch.cat(labels), torch.cat(costs)

    # ---- training loop ----
    def train(self):
        cfg = self.mentor_config
        self.t_start = time.time()
        rounds = int(cfg.rounds)
        if self.A_hat is None:
            stats = self.invert()
            if int(cfg.get('synergies', {}).get('dims', 0)) > 0:
                stats.update(self.fit_synergies())
            self.data_s = self.S[self.valid].clone()
            self.data_a = getattr(self, 'A_labels', self.A_hat)[self.valid].clone()
            fit_loss = self.fit(int(cfg.fit.init_steps), float(cfg.fit.init_lr), 0)
            self.epoch = 0
            self._log_round({**stats, 'train/fit_loss': fit_loss, 'train/dataset_size': self.data_s.shape[0]})
        if self.full_state:
            from .mentor_rewarped import make_state_teacher_class

            TeacherCls = make_state_teacher_class(RejoinTeacher)
            unsupported = {
                'augment.replay_batches': int(cfg.get('augment', {}).get('replay_batches', 0)),
                'augment.perturb_batches': int(cfg.get('augment', {}).get('perturb_batches', 0)),
                'gate.auto_noise': float(cfg.gate.get('auto_noise', 0.0)),
                'gate.advantage': float(cfg.gate.get('advantage', 0.0)),
                'bootstrap.max_segments': int(cfg.get('bootstrap', {}).get('max_segments', 0)),
                'bootstrap.teacher_steps': int(cfg.get('bootstrap', {}).get('teacher_steps', 0)),
            }
            bad = [k for k, v in unsupported.items() if v > 0]
            if bad:
                raise ValueError(f'full-state MENTOR has no observation-to-state map, so these need to be off: {bad}')
        else:
            TeacherCls = RejoinTeacher
        self.teacher = TeacherCls(
            self.spec,
            self.S,
            self.S1,
            self.A_hat,
            self.lengths,
            int(cfg.teacher.batch),
            self.device,
            self.counter,
            cfg.teacher,
        )
        gate_tau, gate_q = float(cfg.gate.tau), float(cfg.gate.quantile)
        self.pending_advantage = []
        self.teacher.anchor_to(self.policy)
        auto_noise = float(cfg.gate.get('auto_noise', 0.0))
        if auto_noise > 0:  # tau = factor x median teacher cost on states a few noisy demo-replay steps away
            n_cal = int(cfg.gate.get('auto_batches', 3))
            xs = self.replay_demo_states(n_cal * int(cfg.teacher.batch), int(cfg.gate.get('auto_steps', 4)), auto_noise)
            _, _, c_cal = self.label(xs, n_cal)
            gate_tau, gate_q = float(cfg.gate.get('auto_factor', 2.0)) * c_cal.median().item(), 0.0
            self.pending_advantage = []
            print(f'[MENTOR] calibrated gate: tau = {gate_tau:.4g} (median calibration cost {c_cal.median().item():.4g})')
            self.writer.add(
                self.agent_steps, {'gate/auto_tau': gate_tau, 'gate/calibration_cost_median': c_cal.median().item()}
            )
            self.writer.write()
        if int(cfg.get('bootstrap', {}).get('teacher_steps', 0)) > 0 and self.round == 0:
            self.teacher_demonstrations()
        while self.round < rounds:
            if self.job_clock is not None:
                _, safe_stop = self.job_clock.step(check_safe_stop=True)
                if safe_stop:
                    print('Not enough time left for another round. Exiting cleanly.')
                    break
            self.round += 1
            self.epoch = self.round
            t0 = time.time()
            visited = self.collect()
            t_roll = time.time() - t0
            x, a0, cost = self.label(visited)
            library_stats = self.bootstrap_library(cost)
            n_perturb = int(cfg.get('augment', {}).get('perturb_batches', 0))
            if n_perturb > 0:  # teacher labels on noisy copies of demo states (DART-style coverage)
                xp, ap, cp = self.label(self.perturbed_demo_states(n_perturb * int(cfg.teacher.batch)), n_perturb)
                x, a0, cost = torch.cat([x, xp]), torch.cat([a0, ap]), torch.cat([cost, cp])
            aug = cfg.get('augment', {})
            n_replay = int(aug.get('replay_batches', 0))
            if n_replay > 0:  # teacher labels on states reached by noisy replays of the demo actions
                xs = self.replay_demo_states(
                    n_replay * int(cfg.teacher.batch), int(aug.get('replay_steps', 4)), float(aug.get('replay_noise', 0.2))
                )
                xr, ar, cr = self.label(xs, n_replay)
                x, a0, cost = torch.cat([x, xr]), torch.cat([a0, ar]), torch.cat([cost, cr])
            threshold = max(gate_tau, cost.quantile(gate_q).item()) if gate_q > 0 else gate_tau
            keep = cost <= threshold
            adv_stats = {}
            if self.pending_advantage:  # keep a label only where the teacher's plan beats the policy
                errs = torch.cat(self.pending_advantage)
                beats = errs[:, 0] <= float(cfg.gate.advantage) * errs[:, 1]
                keep = keep & beats
                adv_stats = {
                    'gate/teacher_beats_policy': beats.float().mean().item(),
                    'gate/teacher_err_q50': errs[:, 0].median().item(),
                    'gate/policy_err_q50': errs[:, 1].median().item(),
                }
            self.pending_advantage = []
            self.data_s = torch.cat([self.data_s, x[keep]])
            self.data_a = torch.cat([self.data_a, a0[keep]])
            t_teach = time.time() - t0 - t_roll
            torch.cuda.empty_cache()
            t1 = time.time()
            fit_loss = self.fit(int(cfg.fit.steps), float(cfg.fit.lr), self.round)
            q = torch.quantile(cost, torch.tensor([0.1, 0.5, 0.9], device=self.device))
            self._log_round(
                {
                    'train/teacher_cost_q10': q[0].item(),
                    'train/teacher_cost_q50': q[1].item(),
                    'train/teacher_cost_q90': q[2].item(),
                    'train/gate_threshold': threshold,
                    'train/kept_fraction': keep.float().mean().item(),
                    'train/dataset_size': self.data_s.shape[0],
                    'train/fit_loss': fit_loss,
                    **library_stats,
                    **adv_stats,
                    'time/rollout': t_roll,
                    'time/teacher': t_teach,
                    'time/fit': time.time() - t1,
                }
            )
        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

    def _log_round(self, train_stats):
        t0 = time.time()
        returns, lengths = self.evaluate(int(self.mentor_config.eval_episodes_per_env))
        mean_ret = float(np.mean(returns))
        metrics = {
            'round': self.round,
            **train_stats,
            **self._sim_stats(),
            'time/eval': time.time() - t0,
            'time/wall': self._wall(),
            'eval_scores/num_episodes': len(returns),
            'eval_scores/episode_rewards': mean_ret,
            'eval_scores/episode_lengths': float(np.mean(lengths)),
            'eval_scores/normalized_return': mean_ret / self.expert_return,
        }
        self.writer.add(self.agent_steps, metrics)
        self.writer.write()
        print(
            f'[MENTOR] round {self.round}: return {mean_ret:.0f} ({mean_ret / self.expert_return:.3f} of expert), '
            f'len {np.mean(lengths):.0f}, sim steps {metrics["sim_steps/total"]:.3g}, wall {metrics["time/wall"]:.0f}s'
        )
        self._write_scores(returns, lengths)
        self._checkpoint_save(mean_ret)

    def _write_scores(self, returns, lengths):
        scores = {
            'epoch': self.epoch,
            'mini_epoch': self.mini_epoch,
            'agent_steps': self.agent_steps,
            **self._sim_stats(),
            'expert_return': self.expert_return,
            'eval_scores/num_episodes': len(returns),
            'eval_scores/episode_rewards': returns,
            'eval_scores/episode_lengths': lengths,
        }
        json.dump(scores, open(os.path.join(self.logdir, 'scores.json'), 'w'), indent=4)

    @torch.no_grad()
    def evaluate(self, episodes_per_env=2):
        """Deterministic policy, separate env (stochastic init, early termination as in training).

        Every env contributes its first ``episodes_per_env`` episodes, as in SHAC/FOCUS eval.
        """
        if not hasattr(self, 'eval_env') and self.full_state:
            self.eval_env = self.spec.make_env(self.num_actors, self.device, seed=self.seed + 12345, grad=False)
        if not hasattr(self, 'eval_env'):
            env_cfg = self.full_cfg.task.env
            self.eval_env = self.spec.make_env(
                self.num_actors,
                self.device,
                seed=self.seed + 12345,
                stochastic_init=bool(env_cfg.get('stochastic_init', True)),
                early_termination=bool(env_cfg.get('early_termination', True)),
            )
        env = self.eval_env
        self.policy.eval()
        obs = self._state_obs(self._convert_obs(env.reset()))
        n = self.num_actors
        ret = torch.zeros(n, device=self.device)
        length = torch.zeros(n, dtype=torch.long, device=self.device)
        count = torch.zeros(n, dtype=torch.long, device=self.device)
        returns, lengths = [], []
        max_steps = episodes_per_env * env.episode_length + 10
        for _ in range(max_steps):
            with torch.no_grad():
                o, r, d, _ = env.step(self.env_action(self.policy(obs)))
            obs = self._state_obs(self._convert_obs(o)).clone()
            ret += r
            length += 1
            for i in d.nonzero(as_tuple=False).flatten().tolist():
                if count[i] < episodes_per_env:
                    returns.append(ret[i].item())
                    lengths.append(int(length[i].item()))
                count[i] += 1
                ret[i] = 0.0
                length[i] = 0
            if bool((count >= episodes_per_env).all()):
                break
        return returns, lengths

    def eval(self):
        returns, lengths = self.evaluate(int(self.mentor_config.eval_episodes_per_env))
        mean_ret = float(np.mean(returns))
        metrics = {
            'eval_scores/num_episodes': len(returns),
            'eval_scores/episode_rewards': mean_ret,
            'eval_scores/episode_lengths': float(np.mean(lengths)),
            'eval_scores/normalized_return': mean_ret / self.expert_return,
        }
        print(metrics)
        self.writer.add(self.agent_steps, metrics)
        self.writer.write()
        self._write_scores(returns, lengths)

    def set_train(self):
        self.policy.train()

    def set_eval(self):
        self.policy.eval()

    def save(self, f):
        ckpt = {
            'epoch': self.epoch,
            'round': self.round,
            'agent_steps': self.agent_steps,
            'policy': self.policy.state_dict(),
            'A_hat': self.A_hat,
            'synergies': (self.spec.mean, self.spec.basis, self.spec.scale) if isinstance(self.spec, SynergySpec) else None,
            'data_s': self.data_s,
            'data_a': self.data_a,
            'counter': self.counter.state_dict(),
            'stage1_steps': self.stage1_steps,
            'wall': self._wall(),
        }
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=''):
        ckpt = torch.load(f, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.epoch, self.round, self.agent_steps = ckpt['epoch'], ckpt['round'], ckpt['agent_steps']
        self.A_hat, self.data_s, self.data_a = ckpt['A_hat'], ckpt['data_s'], ckpt['data_a']
        if ckpt.get('synergies') is not None:
            self.use_synergies(*ckpt['synergies'])
        self.counter.load_state_dict(ckpt['counter'])
        self.stage1_steps = ckpt['stage1_steps']
        self.wall_offset = ckpt.get('wall', 0.0)
        self.policy.eval()
