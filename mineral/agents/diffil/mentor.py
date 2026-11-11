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
from .mentor_sim import OneStep, SimCounter, get_spec


class MentorPolicy(nn.Module):
    """Deterministic MLP policy, tanh output, input standardised with the demo statistics."""

    def __init__(self, obs_dim, act_dim, mean, std, units=(256, 256)):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clone())
        layers, d = [], obs_dim
        for u in units:
            layers += [nn.Linear(d, u), nn.ELU()]
            d = u
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.tanh(self.net((obs - self.mean) / self.std))


@torch.no_grad()
def lm_invert(step, s, s1, w, iters=5, starts=1, h=1e-2, lam0=1e-2, reg=0.0, param='box', generator=None):
    """Batched Levenberg-Marquardt for min_a ||w * (step(s, a) - s1)||^2 + reg ||a||^2 with |a| <= 1.

    Forward-difference Jacobian (act_dim + 1 forward steps per iteration). The first start
    is a = 0, later ones are uniform in (-1, 1); the best solution per sample is kept.
    ``param='box'`` clips each step to [-1, 1]; ``param='tanh'`` optimises u with a = tanh(u),
    which avoids the clipping distortion when many actions sit at a bound. ``reg > 0`` picks
    one consistent (minimum-norm) solution when the actions outnumber what the next state
    determines, as for muscle-driven bodies (SNU: 152 muscles, 18 dofs).
    Returns actions (n, act_dim) and the residual part ||w * (step(s, a) - s1)||^2 (n,).
    """
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
        total = cost + reg * (a**2).sum(-1)
        lam = torch.full_like(cost, lam0)
        for _ in range(iters):
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
            H = J.transpose(1, 2) @ J + reg * torch.diag_embed(da**2)
            g = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1) + reg * a * da
            A = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-6 if tanh else 1e-12))
            x_new = x - torch.linalg.solve(A, g.unsqueeze(-1)).squeeze(-1)
            x_new = x_new if tanh else x_new.clamp(-1, 1)
            r_new = residual(x_new)
            c_new = (r_new**2).sum(-1)
            t_new = c_new + reg * (to_action(x_new) ** 2).sum(-1)
            ok = t_new < total
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
        self.offsets = torch.tensor(list(cfg.phase_offsets), device=device)
        N, T, D = S.shape
        self.N, self.T, self.D = N, T, D
        valid = torch.arange(T, device=device)[None] < lengths[:, None]
        self.w = 1.0 / S[valid].std(0).clamp_min(1e-6)
        self.S_flat = (S * self.w).reshape(-1, D)
        # admissible window starts: a full K-step window plus margin for the phase offsets
        t = torch.arange(T, device=device)[None]
        margin = int(self.offsets.max().clamp_min(0).item()) + 2
        self.window_ok = (t >= 3) & (t <= (lengths[:, None] - self.K - margin))
        if not self.window_ok.any():
            raise ValueError(f'demonstrations too short for a rejoin horizon of {self.K}')
        n_plan = batch * self.m
        self.sim_all = OneStep(spec, n_plan, device, counter)
        self.sim_one = OneStep(spec, batch, device, counter) if self.screen else None

    def windows(self, x):
        d = torch.cdist(x * self.w, self.S_flat).view(-1, self.N, self.T)
        d = d.masked_fill(~self.window_ok[None], float('inf'))
        dmin, jmin = d.min(-1)
        top = dmin.topk(self.m, dim=1, largest=False).indices
        return top, jmin.gather(1, top)

    def a_ref(self, traj, j):
        return torch.stack([self.A_hat[traj, j + k] for k in range(self.K)], 1)

    def rollout(self, sim, x, a, grad):
        y = x
        for k in range(self.K):
            y = sim(y, a[:, k], grad=grad)
        return y

    def cost(self, y, a, traj, j, a_ref):
        idx = j[:, None] + self.K - 1 + self.offsets[None]
        idx = torch.minimum(idx.clamp_min(0), (self.lengths[traj] - 1)[:, None])
        target = self.S1[traj[:, None], idx]
        err = (((y[:, None] - target) * self.w) ** 2).sum(-1).min(-1).values
        return err + self.lam * ((a - a_ref) ** 2).sum((-1, -2))

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
            if self.iters == 0:
                return a_ref[:, 0], c0.view(n, m)[ar, sel]
            best_a, best = self.optimise(self.sim_one, x, traj, jj, a_ref)
            return best_a[:, 0], best
        best_a, best = self.optimise(self.sim_all, xr, traj_all, j_all, a_ref_all)
        sel = best.view(n, m).argmin(1)
        return best_a.view(n, m, self.K, -1)[ar, sel][:, 0], best.view(n, m)[ar, sel]


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
        self.spec = get_spec(self.env_name)
        self.counter = SimCounter()  # Stage I + teacher simulator transitions
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        # --- Demonstrations (state only) ---
        demos = get_demos(self.device, **cfg.demos)
        demos.pop('act')  # never read: state-only imitation
        self.expert_return = float(demos['expert_return'])
        D = self.spec.obs_dim
        obs, next_obs = demos['obs'], demos['next_obs']
        obs = obs['obs'] if isinstance(obs, dict) else obs
        next_obs = next_obs['obs'] if isinstance(next_obs, dict) else next_obs
        self.S = obs[..., :D].float().contiguous()
        self.S1 = next_obs[..., :D].float().contiguous()
        self.lengths = self._lengths(demos['done'])
        T = self.S.shape[1]
        self.valid = torch.arange(T, device=self.device)[None] < self.lengths[:, None]
        S_valid = self.S[self.valid]
        self.w_step = 1.0 / (self.S1[self.valid] - S_valid).std(0).clamp_min(1e-8)

        # --- Policy ---
        units = tuple(self.network_config.get('units', [256, 256]))
        self.policy = MentorPolicy(D, self.spec.act_dim, S_valid.mean(0), S_valid.std(0).clamp_min(1e-6), units)
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

    def get_actions(self, obs, sample: bool = False):
        return self.policy(self._state_obs(obs))

    # ---- Stage I ----
    def invert(self):
        cfg = self.mentor_config.invert
        batch = int(cfg.batch)
        s, s1 = self.S[self.valid], self.S1[self.valid]
        n = s.shape[0]
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
            )
            acts.append(a[:k])
            costs.append(c[:k])
        del step
        torch.cuda.empty_cache()
        self.stage1_steps = self.counter.total - before
        a_flat, cost = torch.cat(acts), torch.cat(costs)
        A_hat = torch.zeros(*self.S.shape[:2], self.spec.act_dim, device=self.device)
        A_hat[self.valid] = a_flat
        self.A_hat = A_hat
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
    def fit(self, steps, lr, round_idx):
        cfg = self.mentor_config.fit
        opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=float(cfg.lr_min))
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed * 100 + round_idx)
        bs = int(cfg.batch_size)
        self.policy.train()
        for _ in range(steps):
            idx = torch.randint(0, self.data_s.shape[0], (bs,), device=self.device, generator=g)
            loss = ((self.policy(self.data_s[idx]) - self.data_a[idx]) ** 2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
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
        visited = []
        for t in range(T):
            if reset_every > 0 and t % reset_every == 0 and n_reset > 0:
                ids = torch.arange(n_reset, device=self.device)
                self.env.reset(env_ids=ids)
                self.obs = self._state_obs(self._convert_obs(self.env.obs_buf.clone()))
            a = self.policy(self.obs)
            if noise > 0:
                a = (a + noise * torch.randn(a.shape, device=self.device, generator=self.generator)).clamp(-1, 1)
            visited.append(self.obs.clone())
            obs, rew, done, info = self.env.step(a)
            self.obs = self._state_obs(self._convert_obs(obs)).clone()
            self.agent_steps += self.num_actors
        return torch.cat(visited)

    def label(self, visited):
        cfg = self.mentor_config.teacher
        batch, n_batches = int(cfg.batch), int(cfg.batches_per_round)
        need = batch * n_batches
        if visited.shape[0] < need:
            raise ValueError(f'rollouts give {visited.shape[0]} states, the teacher needs {need}; raise rollout.length')
        perm = torch.randperm(visited.shape[0], device=self.device, generator=self.generator)[:need]
        xs, labels, costs = [], [], []
        for c0 in range(0, need, batch):
            x = visited[perm[c0 : c0 + batch]]
            a0, c = self.teacher.label(x)
            xs.append(x)
            labels.append(a0.detach())
            costs.append(c.detach())
        return torch.cat(xs), torch.cat(labels), torch.cat(costs)

    # ---- training loop ----
    def train(self):
        cfg = self.mentor_config
        self.t_start = time.time()
        rounds = int(cfg.rounds)
        if self.A_hat is None:
            stats = self.invert()
            self.data_s = self.S[self.valid].clone()
            self.data_a = self.A_hat[self.valid].clone()
            fit_loss = self.fit(int(cfg.fit.init_steps), float(cfg.fit.init_lr), 0)
            self.epoch = 0
            self._log_round({**stats, 'train/fit_loss': fit_loss, 'train/dataset_size': self.data_s.shape[0]})
        self.teacher = RejoinTeacher(
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
            threshold = max(gate_tau, cost.quantile(gate_q).item()) if gate_q > 0 else gate_tau
            keep = cost <= threshold
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
        obs = self._state_obs(env.reset())
        n = self.num_actors
        ret = torch.zeros(n, device=self.device)
        length = torch.zeros(n, dtype=torch.long, device=self.device)
        count = torch.zeros(n, dtype=torch.long, device=self.device)
        returns, lengths = [], []
        max_steps = episodes_per_env * env.episode_length + 10
        for _ in range(max_steps):
            o, r, d, _ = env.step(self.policy(obs))
            obs = self._state_obs(o).clone()
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
        self.counter.load_state_dict(ckpt['counter'])
        self.stage1_steps = ckpt['stage1_steps']
        self.wall_offset = ckpt.get('wall', 0.0)
        self.policy.eval()
