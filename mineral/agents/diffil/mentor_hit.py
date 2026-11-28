"""TUTOR teachers that learn from every simulated segment: each one is an exact demonstration of reaching where it
landed (hindsight). Two teachers share the segment buffer:

hit       hindsight inverse teacher: goal-conditioned inverse model g(x0, xK) -> plan (K x A); hit_head=mse
          (conditional mean plan) or flow (conditional flow matching, sampled from a fixed latent: one deterministic
          plan per (x0, xK)). Label: p_0 = g(x, s*), then hit_iters times: roll p, it lands at y,
          p <- clip(p + alpha (g(x, s*) - g(x, y))), alpha the best of hit_alphas (one rollout each; the model's
          error at y is known exactly since p reaches y, so the difference cancels its local bias). The label is the
          plan with the lowest rejoin cost among the references and every rolled plan.
lmmodel   Levenberg-Marquardt on the plan (as lmplan) with the Jacobian of a learned forward model
          f(x0, plan) -> w (xK - x0) (network gradients, never simulator gradients) and the simulator's residuals:
          lm_model_jac=broyden (model Jacobian at the start, then Broyden updates from the true rollouts) or every
          (the model re-linearised at every iterate). Before the model is trained, the first Jacobian is by forward
          differences, as lmplan.

Data, all already simulated: the demonstration windows with the Stage I actions (s_j, a_hat_j..j+K-1, s'_{j+K-1});
every K-step window without a reset or a fall of each round's learner rollouts (TUTOR.collect hands them over);
every rollout the teacher itself makes. Per-round diagnostics go to the log (TUTOR._teacher_iter_stats).
Cost per label (forward steps; plus screening num_windows K and anchor K - 1):
  hit: K (references) + K + hit_iters len(hit_alphas) K + hit_explore K
  lmmodel: K (references) + K + iters K (+ (K A + 1) K for a forward-difference Jacobian while the model is untrained)
"""

import torch
import torch.nn as nn

from .mentor_gf import LMPlanTeacher, _GFTeacher


class _MLP(nn.Module):
    def __init__(self, i, o, units=(512, 512, 512)):
        super().__init__()
        layers, d = [], i
        for u in units:
            layers += [nn.Linear(d, u), nn.SiLU()]
            d = u
        layers.append(nn.Linear(d, o))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Segments:
    """Mixin: buffer of K-step segments (x0, xK, plan) and a network fitted on it (subclasses define loss())."""

    def init_segments(self, cfg, n_in, n_out):
        self.cap = int(cfg.get('hit_buffer', 1_000_000))
        self.round_steps = int(cfg.get('hit_round_steps', 2000))
        self.batch_steps = int(cfg.get('hit_batch_steps', 50))
        self.min_data = int(cfg.get('hit_min_data', 5000))
        valid = torch.arange(self.T, device=self.device)[None] < self.lengths[:, None]
        self.mu, self.sd = self.S[valid].mean(0), self.S[valid].std(0).clamp_min(1e-6)
        self.net = _MLP(n_in, n_out).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(cfg.get('hit_lr', 1e-3)))
        self.buf, self.trained, self.last_loss = None, False, float('nan')
        self.stats = []
        S, S1, A_hat, lengths, K = self.S, self.S1, self.A_hat, self.lengths, self.K
        jj = torch.arange(S.shape[1] - K, device=self.device)
        ok = (jj[None] + K) <= lengths[:, None]
        tr, j0 = ok.nonzero(as_tuple=True)
        idx = j0[:, None] + torch.arange(K, device=self.device)[None]
        self.add(S[tr, j0], S1[tr, j0 + K - 1], A_hat[tr[:, None], idx])

    @torch.no_grad()
    def add(self, x0, xK, plan):
        plan = plan.reshape(plan.shape[0], -1)
        ok = torch.isfinite(x0).all(-1) & torch.isfinite(xK).all(-1) & torch.isfinite(plan).all(-1)
        new = (x0[ok], xK[ok], plan[ok].clamp(-1, 1))
        self.buf = new if self.buf is None else tuple(torch.cat([b, c])[-self.cap :] for b, c in zip(self.buf, new))

    @torch.no_grad()
    def observe_rollout(self, roll):
        obs, act, brk = roll['obs'], roll['act'], roll['broken']  # [T, n, D], [T, n, A], [T, n]
        K, xs, ys, ps = self.K, [], [], []
        for t in range(obs.shape[0] - K):
            ok = ~brk[t : t + K + 1].any(0)  # no fall in t..t+K-1, no reset up to t+K
            if ok.any():
                xs.append(obs[t][ok])
                ys.append(obs[t + K][ok])
                ps.append(act[t : t + K].transpose(0, 1)[ok])
        if xs:
            self.add(torch.cat(xs), torch.cat(ys), torch.cat(ps))
        self.fit(self.round_steps)

    def fit(self, steps, bs=1024):
        if self.buf is None or self.buf[0].shape[0] < self.min_data or steps <= 0:
            return
        N = self.buf[0].shape[0]
        self.net.train()
        with torch.enable_grad():
            for _ in range(steps):
                i = torch.randint(0, N, (bs,), device=self.device, generator=self.gen)
                loss = self.loss(self.buf[0][i], self.buf[1][i], self.buf[2][i].view(-1, self.K, self.A))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        self.net.eval()
        self.trained, self.last_loss = True, loss.item()

    def round_stats(self):
        """Mean of the per-batch diagnostics since the last call, plus buffer size and model loss."""
        out = {'teacher/buffer': float(self.buf[0].shape[0]) if self.buf is not None else 0.0,
               'teacher/model_loss': self.last_loss}
        if self.stats:
            for k in self.stats[0]:
                v = [s[k] for s in self.stats if k in s]
                out[f'teacher/{k}'] = sum(v) / len(v)
        self.stats = []
        return out


class HindsightInverseTeacher(_Segments, _GFTeacher):
    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        _GFTeacher.__init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        self.head = str(cfg.get('hit_head', 'mse'))
        if self.head not in ('mse', 'flow'):
            raise ValueError(f'teacher.hit_head must be mse or flow, not {self.head!r}')
        self.hit_iters = int(cfg.get('hit_iters', 3))
        self.alphas = [float(a) for a in cfg.get('hit_alphas', [1.0])]
        self.explore = int(cfg.get('hit_explore', 4))
        self.sigma = float(cfg.get('sample_noise', 0.1))
        self.flow_steps = int(cfg.get('hit_flow_steps', 10))
        D, KA = self.D, self.K * self.A
        self.init_segments(cfg, 2 * D if self.head == 'mse' else 2 * D + KA + 1, KA)

    def cond(self, x0, xK):
        return torch.cat([(x0 - self.mu) / self.sd, (xK - x0) * self.w], -1)

    def loss(self, x0, xK, plan):
        c, plan = self.cond(x0, xK), plan.reshape(plan.shape[0], -1)
        if self.head == 'mse':
            return ((torch.tanh(self.net(c)) - plan) ** 2).mean()
        z0 = torch.randn_like(plan)
        t = torch.rand(plan.shape[0], 1, device=plan.device)
        zt = (1 - t) * z0 + t * plan
        return ((self.net(torch.cat([c, zt, t], -1)) - (plan - z0)) ** 2).mean()

    @torch.no_grad()
    def predict(self, x0, xK):
        c = self.cond(x0, xK)
        if self.head == 'mse':
            out = torch.tanh(self.net(c))
        else:  # Euler integration of the flow from a fixed latent (zero)
            z = torch.zeros(x0.shape[0], self.K * self.A, device=x0.device)
            h = 1.0 / self.flow_steps
            for k in range(self.flow_steps):
                t = torch.full((x0.shape[0], 1), k * h, device=x0.device)
                z = z + h * self.net(torch.cat([c, z, t], -1))
            out = z.clamp(-1, 1)
        return out.view(-1, self.K, self.A)

    @torch.no_grad()
    def optimise(self, sim, x, traj, j, a_ref):
        tgt, e0 = self.fixed_target(x, a_ref, traj, j)
        y_ref, out_ref = self._ref_rollout
        self.add(x, y_ref, a_ref)
        best_a, best_c = a_ref.clone(), self.cost((y_ref, out_ref), a_ref, traj, j, a_ref, x)
        c_ref = best_c.clone()
        e0n = e0.norm(dim=-1).clamp_min(1e-9)
        st = {'trained': float(self.trained)}
        if self.trained:
            g_star = self.predict(x, tgt)
            p = g_star.clone()
            y, out = self.rollout(self.sim_base, x, p, grad=False)
            self.add(x, y, p)
            c = self.cost((y, out), p, traj, j, a_ref, x)
            st['model_err_ratio'] = (((y - tgt) * self.w).norm(dim=-1) / e0n).median().item()
            ar = torch.arange(x.shape[0], device=x.device)
            for it in range(self.hit_iters):
                better = c < best_c
                best_a = torch.where(better[:, None, None], p, best_a)
                best_c = torch.where(better, c, best_c)
                step = g_star - self.predict(x, y)
                cand = []
                for a in self.alphas:  # step-size search: one rollout per alpha, keep the best per state
                    q = (p + a * step).clamp(-1, 1)
                    yq, oq = self.rollout(self.sim_base, x, q, grad=False)
                    self.add(x, yq, q)
                    cand.append((self.cost((yq, oq), q, traj, j, a_ref, x), q, yq))
                cs = torch.stack([cc for cc, _, _ in cand], 1)
                k = cs.argmin(1)
                c = cs[ar, k]
                p = torch.stack([q for _, q, _ in cand], 1)[ar, k]
                y = torch.stack([yy for _, _, yy in cand], 1)[ar, k]
            better = c < best_c
            best_a = torch.where(better[:, None, None], p, best_a)
            best_c = torch.where(better, c, best_c)
            st['final_err_ratio'] = (((y - tgt) * self.w).norm(dim=-1) / e0n).median().item()
            st['label_is_ref'] = (best_c >= c_ref).float().mean().item()
            st['cost_ratio'] = (best_c / c_ref.clamp_min(1e-9)).median().item()
        for _ in range(self.explore):  # buffer only
            q = (best_a + self.sigma * torch.randn(best_a.shape, device=x.device, generator=self.gen)).clamp(-1, 1)
            y, _ = self.rollout(self.sim_base, x, q, grad=False)
            self.add(x, y, q)
        self.fit(self.batch_steps)
        self.stats.append(st)
        self.iters_used.append(self.hit_iters)
        return best_a, best_c


class ModelLMTeacher(_Segments, LMPlanTeacher):
    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        LMPlanTeacher.__init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        self.model_jac = str(cfg.get('lm_model_jac', 'broyden'))
        if self.model_jac not in ('broyden', 'every'):
            raise ValueError(f'teacher.lm_model_jac must be broyden or every, not {self.model_jac!r}')
        self.jac = 'fd' if self.model_jac == 'every' else 'broyden'  # LMPlanTeacher: recompute every iteration or once
        self.init_segments(cfg, self.D + self.K * self.A, self.D)
        self._n_fd = 0
        # lm_sobolev > 0: also fit the model's differences to secant pairs, f(x, a1) - f(x, a0) = w (y1 - y0), from
        # consecutive LM rollouts of the same states (exact, free); relative squared error, weight lm_sobolev
        self.sobolev = float(cfg.get('lm_sobolev', 0.0))
        self.pairs, self._prev = None, None

    def loss(self, x0, xK, plan):
        inp = torch.cat([(x0 - self.mu) / self.sd, plan.reshape(plan.shape[0], -1)], -1)
        return ((self.net(inp) - (xK - x0) * self.w) ** 2).mean()

    def jacobian(self, x, flat, y):
        """d(w xK)/d plan (n, D, KA): the model's (network gradient) once it is trained, else forward differences."""
        if not self.trained:
            self._n_fd += 1
            return LMPlanTeacher.jacobian(self, x, flat, y)
        from torch.func import jacrev, vmap

        z = (x - self.mu) / self.sd
        f = lambda zi, pi: self.net(torch.cat([zi, pi], -1)[None])[0]
        with torch.enable_grad():
            J = vmap(jacrev(f, argnums=1))(z, flat.detach())
        return J.detach()

    @torch.no_grad()
    def residual(self, x, flat, traj, j, a_ref_flat):
        y, r_state, r_act, cost = LMPlanTeacher.residual(self, x, flat, traj, j, a_ref_flat)
        self.add(x, y, flat)  # every LM rollout is a segment
        if self.sobolev > 0:  # secant pair with the previous rollout of the same states
            if self._prev is not None and self._prev[0] is x:
                _, a0, y0 = self._prev
                d = (flat - a0).abs().amax(-1) > 1e-6
                if d.any():
                    new = (x[d], a0[d], flat[d], ((y - y0) * self.w)[d])
                    self.pairs = new if self.pairs is None else tuple(
                        torch.cat([b, c])[-self.cap // 2 :] for b, c in zip(self.pairs, new))
            self._prev = (x, flat.clone(), y.clone())
        return y, r_state, r_act, cost

    def fit(self, steps, bs=1024):
        if self.sobolev <= 0 or self.pairs is None:
            return super().fit(steps, bs)
        if self.buf is None or self.buf[0].shape[0] < self.min_data or steps <= 0:
            return
        N, M = self.buf[0].shape[0], self.pairs[0].shape[0]
        self.net.train()
        with torch.enable_grad():
            for _ in range(steps):
                i = torch.randint(0, N, (bs,), device=self.device, generator=self.gen)
                loss = self.loss(self.buf[0][i], self.buf[1][i], self.buf[2][i].view(-1, self.K, self.A))
                k = torch.randint(0, M, (bs,), device=self.device, generator=self.gen)
                x, a0, a1, dy = (t[k] for t in self.pairs)
                z = (x - self.mu) / self.sd
                pred = self.net(torch.cat([z, a1], -1)) - self.net(torch.cat([z, a0], -1))
                sec = ((pred - dy) ** 2).sum(-1).mean() / (dy**2).sum(-1).mean().clamp_min(1e-12)
                total = loss + self.sobolev * sec
                self.opt.zero_grad()
                total.backward()
                self.opt.step()
        self.net.eval()
        self.trained, self.last_loss = True, loss.item()
        self.last_secant = sec.item()

    def round_stats(self):
        out = super().round_stats()
        if self.sobolev > 0:
            out['teacher/secant_rel_err'] = float(getattr(self, 'last_secant', float('nan')))
            out['teacher/secant_pairs'] = float(self.pairs[0].shape[0]) if self.pairs is not None else 0.0
        return out

    @torch.no_grad()
    def optimise(self, sim, x, traj, j, a_ref):
        self._n_fd = 0
        self.fixed_target(x, a_ref, traj, j)
        c_ref = self.cost(self._ref_rollout, a_ref, traj, j, a_ref, x)
        a, c = LMPlanTeacher.optimise(self, sim, x, traj, j, a_ref)
        self.stats.append({'trained': float(self.trained), 'fd_jacobians': float(self._n_fd),
                           'cost_ratio': (c / c_ref.clamp_min(1e-9)).median().item(),
                           'label_is_ref': (c >= c_ref).float().mean().item()})
        self.fit(self.batch_steps)
        return a, c
