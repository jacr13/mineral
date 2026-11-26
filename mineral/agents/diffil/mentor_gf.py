"""TUTOR: rejoin teachers without simulator gradients (forward steps only).

Every teacher here keeps RejoinTeacher's windows, screening, anchored warm start and rejoin cost (min over phase
offsets of the weighted terminal error, plus action_reg ||a - a_ref||^2 and the optional effort / smoothness /
barrier terms) and replaces only the optimiser of the K-step open-loop plan. The simulator is called with
grad=False only, so every step is counted as a forward step (counter.fwd) and no backward pass exists.

teacher.kind:
  fdadam  the gradient teacher's own Adam (lr, iters, clamp to [-1, 1], best iterate kept) on a gradient of the plan
          cost estimated from forward steps: fd_mode=full, one-sided forward differences over all K * act_dim plan
          coordinates (the direct test: is it the exact gradient or the optimiser that Ant needs?); fd_mode=rand,
          fd_dirs random Gaussian directions per iteration (SPSA-like, cheaper, noisier).
  lmplan  Levenberg-Marquardt (Gauss-Newton with damping) on the residual [w (x_K(a) - target), sqrt(action_reg)
          (a - a_ref)], target = the best phase offset at the current iterate; Jacobian of x_K by forward
          differences (lm_jacobian=fd) or once by forward differences, then Broyden rank-1 updates (broyden).
  mppi    consistent sampling: Gaussian perturbations of small radius around the current mean, started at the
          anchored warm start; the new mean is the softmin-weighted mean of the perturbations (not the best sample),
          with an extra pull mppi_pull ||u_0 - u_0^warm||^2 of the first action towards the warm start in the cost.
          Neighbouring states then get neighbouring labels.

Cost per label (forward steps; K horizon, A act_dim, I iterations), for the optimiser only (screening adds
num_windows * K, anchoring K - 1):
  fdadam full: I (K A + 1) K + K;  fdadam rand: I (M + 1) K + K;  lmplan fd: I (K A + 2) K + K;
  lmplan broyden: (K A + 1) K + I K + K;  mppi: I (P + 1) K + K.
"""

import torch

from .mentor import RejoinTeacher


class _GFTeacher(RejoinTeacher):
    """Shared machinery: a wide one-step simulator that rolls out many perturbed plans of a batch at once."""

    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        if self.iters_cfg_zero(cfg):
            raise ValueError('gradient-free teachers need teacher.iters > 0')
        self.A = spec.act_dim
        # plans rolled out together per call of the wide simulator: chunk * batch states
        self.chunk = max(1, int(cfg.get('gf_chunk', 16)))
        self.sim_wide = self.make_sim(spec, batch * self.chunk, device, counter)
        self.sim_base = self.sim_one if self.sim_one is not None else self.make_sim(spec, batch, device, counter)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(cfg.get('gf_seed', 0)))

    @staticmethod
    def iters_cfg_zero(cfg):
        return int(cfg.iters) <= 0

    # cost of many candidate plans per state: a (n, P, K, A) -> (n, P)
    @torch.no_grad()
    def costs(self, x, a, traj, j, a_ref, extra=None):
        n, P = a.shape[:2]
        out = []
        for p0 in range(0, P, self.chunk):
            p1 = min(P, p0 + self.chunk)
            c = p1 - p0
            ap = a[:, p0:p1].transpose(0, 1).reshape(c * n, self.K, self.A)  # chunk-major: [p0 states, p0+1 states...]
            rep = lambda t: t.repeat(c, *([1] * (t.dim() - 1)))
            xs, tr, jj, ar = rep(x), rep(traj), rep(j), rep(a_ref)
            if c == self.chunk:
                sim = self.sim_wide
            elif c == 1:
                sim = self.sim_base
            else:  # pad the last chunk to the wide simulator's batch
                pad = self.chunk - c
                xs = torch.cat([xs, x.repeat(pad, 1)])
                ap = torch.cat([ap, a_ref.repeat(pad, 1, 1)])
                tr, jj, ar = torch.cat([tr, traj.repeat(pad)]), torch.cat([jj, j.repeat(pad)]), torch.cat([ar, a_ref.repeat(pad, 1, 1)])
                sim = self.sim_wide
            cc = self.cost(self.rollout(sim, xs, ap, grad=False), ap, tr, jj, ar, xs)
            if extra is not None:
                cc = cc + extra(ap)
            out.append(cc[: c * n].view(c, n).T)
        return torch.cat(out, 1)

    @torch.no_grad()
    def final_states(self, x, a):
        """x_K for candidate plans a (n, P, K, A) -> (n, P, D)."""
        n, P = a.shape[:2]
        out = []
        for p0 in range(0, P, self.chunk):
            p1 = min(P, p0 + self.chunk)
            c = p1 - p0
            ap = a[:, p0:p1].transpose(0, 1).reshape(c * n, self.K, self.A)
            xs = x.repeat(c, 1)
            if c == self.chunk:
                sim = self.sim_wide
            elif c == 1:
                sim = self.sim_base
            else:
                pad = self.chunk - c
                xs = torch.cat([xs, x.repeat(pad, 1)])
                ap = torch.cat([ap, ap[:n].repeat(pad, 1, 1)])
                sim = self.sim_wide
            y, _ = self.rollout(sim, xs, ap, grad=False)
            out.append(y[: c * n].view(c, n, -1).transpose(0, 1))
        return torch.cat(out, 1)

    @torch.no_grad()
    def one_cost(self, x, a, traj, j, a_ref):
        return self.cost(self.rollout(self.sim_base, x, a, grad=False), a, traj, j, a_ref, x)

    @torch.no_grad()
    def fixed_target(self, x, a_ref, traj, j):
        """Roll the references once; the phase offset with the smallest end error fixes the target for this state.
        Returns the target (n, D) and the weighted end error of the references (n, D)."""
        y, _ = self.rollout(self.sim_base, x, a_ref, grad=False)
        tg = self.targets(traj, j, x)
        e = (y[:, None] - tg) * self.w
        k = (e**2).sum(-1).argmin(-1)
        ar = torch.arange(x.shape[0], device=x.device)
        return tg[ar, k], e[ar, k]


class FDAdamTeacher(_GFTeacher):
    """The gradient teacher's Adam on a forward-difference (or random-direction) estimate of the plan-cost gradient."""

    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        self.fd_mode = str(cfg.get('fd_mode', 'full'))
        if self.fd_mode not in ('full', 'rand', 'broyden'):
            raise ValueError(f'teacher.fd_mode must be full, rand or broyden, not {self.fd_mode!r}')
        if self.fd_mode == 'broyden' and (self.effort > 0 or self.smooth > 0 or self.barrier > 0):
            raise ValueError('teacher.fd_mode=broyden supports the rejoin error and action_reg only')
        self.fd_h = float(cfg.get('fd_h', 1e-2))
        self.fd_dirs = int(cfg.get('fd_dirs', 16))

    @torch.no_grad()
    def optimise_broyden(self, x, traj, j, a_ref):
        """Adam on g = 2 J^T r_state + 2 lam (a - a_ref), J = d(w x_K)/da by forward differences once, then Broyden
        updates between consecutive iterates (same Jacobians as lmplan with lm_jacobian=broyden; only the optimiser
        differs)."""
        n, KA = x.shape[0], self.K * self.A
        ar = a_ref.reshape(n, KA)
        lm = self  # LMPlanTeacher's residual / jacobian helpers read only attributes this teacher has
        a = ar.clone().requires_grad_(True)
        opt = torch.optim.Adam([a], lr=self.lr)
        y, rs, ra, cost = LMPlanTeacher.residual(lm, x, a.detach(), traj, j, ar)
        J = LMPlanTeacher.jacobian(lm, x, a.detach(), y)
        best, best_a = cost.clone(), a.detach().clone()
        for it in range(self.iters):
            a.grad = 2 * (J.transpose(1, 2) @ rs[..., None]).squeeze(-1) + 2 * self.lam * (a.detach() - ar)
            prev, y_prev = a.detach().clone(), y
            opt.step()
            a.data.clamp_(-1, 1)
            y, rs, ra, cost = LMPlanTeacher.residual(lm, x, a.detach(), traj, j, ar)
            dx = a.detach() - prev
            dy = (y - y_prev) * self.w
            den = (dx**2).sum(-1, keepdim=True).clamp_min(1e-12)
            J = J + ((dy - (J @ dx[..., None]).squeeze(-1)) / den)[..., None] * dx[:, None]
            better = cost < best
            best = torch.where(better, cost, best)
            best_a[better] = a.detach()[better]
        self.iters_used.append(self.iters)
        out = best_a.view(n, self.K, self.A)
        return out, self.one_cost(x, out, traj, j, a_ref)

    @torch.no_grad()
    def grad(self, x, a, c, traj, j, a_ref):
        n, KA = a.shape[0], self.K * self.A
        flat = a.reshape(n, KA)
        h = self.fd_h
        if self.fd_mode == 'full':
            # one-sided steps away from the bound (as Stage I): -h where a + h > 1
            sign = torch.where(flat + h > 1, -1.0, 1.0)
            eye = torch.eye(KA, device=a.device)
            pert = flat[:, None] + (sign * h)[:, None] * eye[None]  # (n, KA, KA)
            cs = self.costs(x, pert.view(n, KA, self.K, self.A), traj, j, a_ref)
            return ((cs - c[:, None]) / (sign * h)).view(n, self.K, self.A)
        M = self.fd_dirs
        u = torch.randn(n, M, KA, device=a.device, generator=self.gen)
        pert = (flat[:, None] + h * u).clamp(-1, 1)
        du = (pert - flat[:, None]) / h  # the step actually taken after clamping
        cs = self.costs(x, pert.view(n, M, self.K, self.A), traj, j, a_ref)
        # E[(u . g) u] = g for u ~ N(0, I): average of directional derivatives times directions
        g = ((cs - c[:, None]) / h)[:, :, None] * du
        return g.mean(1).view(n, self.K, self.A)

    @torch.no_grad()
    def optimise(self, sim, x, traj, j, a_ref):
        if self.fd_mode == 'broyden':
            return self.optimise_broyden(x, traj, j, a_ref)
        a = a_ref.clone().requires_grad_(True)  # a leaf for torch.optim.Adam only; no autograd graph is ever built
        opt = torch.optim.Adam([a], lr=self.lr)
        best = torch.full((a.shape[0],), float('inf'), device=self.device)
        best_a = a_ref.clone()
        for it in range(self.iters + 1):
            c = self.one_cost(x, a.detach(), traj, j, a_ref)
            better = c < best
            best = torch.where(better, c, best)
            best_a[better] = a.detach()[better].clamp(-1, 1)
            if it == self.iters:
                self.iters_used.append(it)
                break
            a.grad = self.grad(x, a.detach(), c, traj, j, a_ref)
            opt.step()
            a.data.clamp_(-1, 1)
        return best_a, best


class LMPlanTeacher(_GFTeacher):
    """Levenberg-Marquardt on the K-step plan; Jacobian of the final state by forward differences or Broyden."""

    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        self.jac = str(cfg.get('lm_jacobian', 'fd'))
        if self.jac not in ('fd', 'broyden'):
            raise ValueError(f'teacher.lm_jacobian must be fd or broyden, not {self.jac!r}')
        self.fd_h = float(cfg.get('fd_h', 1e-2))
        self.lam0 = float(cfg.get('lm_lambda', 1e-2))
        if self.effort > 0 or self.smooth > 0 or self.barrier > 0:
            raise ValueError('teacher.kind=lmplan supports the rejoin error and action_reg only')

    @torch.no_grad()
    def jacobian(self, x, flat, y):
        n, KA = flat.shape
        h = self.fd_h
        sign = torch.where(flat + h > 1, -1.0, 1.0)
        eye = torch.eye(KA, device=flat.device)
        pert = flat[:, None] + (sign * h)[:, None] * eye[None]
        yp = self.final_states(x, pert.view(n, KA, self.K, self.A))  # (n, KA, D)
        return ((yp - y[:, None]) / (sign * h)[:, :, None]).transpose(1, 2) * self.w[None, :, None]  # (n, D, KA)

    @torch.no_grad()
    def residual(self, x, flat, traj, j, a_ref_flat):
        y, _ = self.rollout(self.sim_base, x, flat.view(-1, self.K, self.A), grad=False)
        tg = self.targets(traj, j, x)  # (n, offsets, D)
        e = ((y[:, None] - tg) * self.w)
        k = (e**2).sum(-1).argmin(-1)
        r_state = e[torch.arange(x.shape[0], device=x.device), k]
        r_act = (self.lam**0.5) * (flat - a_ref_flat)
        cost = (r_state**2).sum(-1) + (r_act**2).sum(-1)
        return y, r_state, r_act, cost

    @torch.no_grad()
    def optimise(self, sim, x, traj, j, a_ref):
        n, KA = x.shape[0], self.K * self.A
        ar = a_ref.reshape(n, KA)
        flat = ar.clone()
        y, rs, ra, cost = self.residual(x, flat, traj, j, ar)
        lam = torch.full_like(cost, self.lam0)
        J = None
        I = torch.eye(KA, device=x.device)[None]
        for it in range(self.iters):
            if J is None or self.jac == 'fd':
                J = self.jacobian(x, flat, y)
            H = J.transpose(1, 2) @ J + self.lam * I
            g = (J.transpose(1, 2) @ rs[..., None]).squeeze(-1) + (self.lam**0.5) * ra
            A_ = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-12))
            new = (flat - torch.linalg.solve(A_, g[..., None]).squeeze(-1)).clamp(-1, 1)
            y_n, rs_n, ra_n, c_n = self.residual(x, new, traj, j, ar)
            if self.jac == 'broyden':  # secant update of the final-state Jacobian, accepted step or not
                dx = new - flat
                dy = (y_n - y) * self.w
                den = (dx**2).sum(-1, keepdim=True).clamp_min(1e-12)
                J = J + ((dy - (J @ dx[..., None]).squeeze(-1)) / den)[..., None] * dx[:, None]
            ok = c_n < cost
            flat = torch.where(ok[:, None], new, flat)
            y = torch.where(ok[:, None], y_n, y)
            rs = torch.where(ok[:, None], rs_n, rs)
            ra = torch.where(ok[:, None], ra_n, ra)
            cost = torch.where(ok, c_n, cost)
            lam = torch.where(ok, (lam * 0.3).clamp_min(1e-9), (lam * 5.0).clamp_max(1e9))
        self.iters_used.append(self.iters)
        a = flat.view(n, self.K, self.A)
        return a, self.one_cost(x, a, traj, j, a_ref)


class MPPITeacher(_GFTeacher):
    """Softmin-weighted mean of small perturbations around the warm start, with a pull of u_0 towards it."""

    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        self.P = int(cfg.get('sample_n', 32))
        self.sigma = float(cfg.get('sample_noise', 0.05))
        self.temp = float(cfg.get('mppi_temp', 0.1))  # temperature, relative to the spread of the sampled costs
        self.pull = float(cfg.get('mppi_pull', 0.0))
        self.antithetic = bool(cfg.get('mppi_antithetic', True))

    @torch.no_grad()
    def optimise(self, sim, x, traj, j, a_ref):
        n = x.shape[0]
        u0w = a_ref[:, 0].clone()
        pull = (lambda ap: self.pull * ((ap[:, 0] - u0w.repeat(ap.shape[0] // n, 1)) ** 2).sum(-1)) if self.pull > 0 else None

        def total(a):
            c = self.one_cost(x, a, traj, j, a_ref)
            return c if pull is None else c + self.pull * ((a[:, 0] - u0w) ** 2).sum(-1)

        mean = a_ref.clone()
        c_mean = total(mean)
        best_a, best_c = mean.clone(), c_mean.clone()
        for _ in range(self.iters):
            half = self.P // 2 if self.antithetic else self.P
            eps = torch.randn(n, half, self.K, self.A, device=x.device, generator=self.gen)
            if self.antithetic:
                eps = torch.cat([eps, -eps], 1)
            cand = (mean[:, None] + self.sigma * eps).clamp(-1, 1)
            cs = self.costs(x, cand, traj, j, a_ref, extra=pull)
            lo = cs.min(1, keepdim=True).values
            spread = (cs.median(1, keepdim=True).values - lo).clamp_min(1e-9)
            wts = torch.softmax(-(cs - lo) / (self.temp * spread), 1)
            mean = (wts[..., None, None] * cand).sum(1).clamp(-1, 1)
            c_mean = total(mean)
            better = c_mean < best_c
            best_a = torch.where(better[:, None, None], mean, best_a)
            best_c = torch.where(better, c_mean, best_c)
        self.iters_used.append(self.iters)
        # report the rejoin cost without the pull, as every other teacher (the gate reads it)
        return best_a, self.one_cost(x, best_a, traj, j, a_ref)


class HindsightTeacher(_GFTeacher):
    """Hindsight tutor (REQUESTS R5a): no optimisation per label, one regression on noisy rollouts.

    From the references a_bar (windows, screening, anchor as the rejoin teacher), with the phase offset fixed by the
    references' own rollout, roll P plans a_bar + eps (eps ~ N(0, sigma^2), clipped) and record the weighted end
    errors e = w (x_K - s*). hs_form=reverse: ridge regression of the plan noise (K * A) on [1, e]; the plan is
    a_bar + intercept (the noise predicted at e = 0). hs_form=forward: ridge regression of e on [1, eps] gives c, J;
    the plan is a_bar + argmin ||c + J d||^2 + action_reg ||a_bar + d - a_ref||^2. hs_passes = 2 re-centres on the
    plan with half the sigma. A verification rollout gives the label's true cost (the gate reads it).
    Cost per label: (P + 2) K for one pass, + (P + 1) K per extra pass, plus screening and anchor.
    """

    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        self.P = int(cfg.get('sample_n', 64))
        self.sigma = float(cfg.get('sample_noise', 0.1))
        self.form = str(cfg.get('hs_form', 'reverse'))
        if self.form not in ('reverse', 'forward'):
            raise ValueError(f'teacher.hs_form must be reverse or forward, not {self.form!r}')
        self.ridge = float(cfg.get('hs_ridge', 1e-2))
        self.passes = int(cfg.get('hs_passes', 1))
        self.stats = []  # per labelled batch: end-error diagnostics (median over states)

    @staticmethod
    def _ridge(X, Y, ridge, free_first=True):
        """Per-state ridge regression: X (n, P, p), Y (n, P, q) -> B (n, p, q); the ridge scales with the mean
        diagonal of X^T X and leaves the intercept column free."""
        XtX = X.transpose(1, 2) @ X
        d = torch.diagonal(XtX, dim1=1, dim2=2)
        scale = d[:, 1:].mean(-1) if free_first else d.mean(-1)
        R = torch.ones(X.shape[-1], device=X.device)
        if free_first:
            R[0] = 0.0
        A_ = XtX + (ridge * scale)[:, None, None] * torch.diag(R)[None] + 1e-9 * torch.eye(X.shape[-1], device=X.device)
        return torch.linalg.solve(A_, X.transpose(1, 2) @ Y)

    @torch.no_grad()
    def optimise(self, sim, x, traj, j, a_ref):
        n, K, A = x.shape[0], self.K, self.A
        tgt, e0 = self.fixed_target(x, a_ref, traj, j)
        centre, sig = a_ref.clone(), self.sigma
        st = {}
        for p in range(self.passes):
            eps = sig * torch.randn(n, self.P, K, A, device=x.device, generator=self.gen)
            cand = (centre[:, None] + eps).clamp(-1, 1)
            eps = (cand - centre[:, None]).reshape(n, self.P, K * A)
            e = (self.final_states(x, cand) - tgt[:, None]) * self.w  # (n, P, D)
            one = torch.ones(n, self.P, 1, device=x.device)
            if self.form == 'reverse':
                B = self._ridge(torch.cat([one, e], -1), eps, self.ridge)
                delta = B[:, 0]
            else:
                B = self._ridge(torch.cat([one, eps], -1), e, self.ridge)
                c, J = B[:, 0], B[:, 1:].transpose(1, 2)  # (n, D), (n, D, KA)
                off = (centre - a_ref).reshape(n, K * A)
                H = J.transpose(1, 2) @ J + self.lam * torch.eye(K * A, device=x.device)[None]
                g = (J.transpose(1, 2) @ c[..., None]).squeeze(-1) + self.lam * off
                delta = -torch.linalg.solve(H + 1e-6 * torch.eye(K * A, device=x.device)[None], g[..., None]).squeeze(-1)
            if p == 0:
                en, e0n = e.norm(dim=-1), e0.norm(dim=-1).clamp_min(1e-9)
                st['min_sample_err_ratio'] = (en.min(1).values / e0n).median().item()
                st['median_sample_err_ratio'] = (en.median(1).values / e0n).median().item()
            centre = (centre + delta.view(n, K, A)).clamp(-1, 1)
            sig = sig / 2
        y, _ = self.rollout(self.sim_base, x, centre, grad=False)
        st['plan_err_ratio'] = (((y - tgt) * self.w).norm(dim=-1) / e0.norm(dim=-1).clamp_min(1e-9)).median().item()
        self.stats.append(st)
        self.iters_used.append(self.passes)
        return centre, self.one_cost(x, centre, traj, j, a_ref)


class _Continuation:
    """G(x, a): one step with a, then K - 1 steps of a continuation (closed-loop policy, or the references open loop).
    Exposes the `spec` and call signature lm_invert expects of a one-step map."""

    def __init__(self, teacher, a_ref, mode):
        self.t, self.a_ref, self.mode = teacher, a_ref, mode
        self.spec = teacher.spec

    @torch.no_grad()
    def __call__(self, s, a, grad=False):
        sim = self.t.sim_base
        y = sim(s, a.clamp(-1, 1), grad=False)
        for k in range(1, self.t.K):
            u = self.t.cont_policy(y).clamp(-1, 1) if self.mode == 'policy' else self.a_ref[:, k]
            y = sim(y, u, grad=False)
        return y

    @torch.no_grad()
    def plan(self, s, a0):
        """The full action sequence G executes from s with first action a0 (n, K, A)."""
        acts, y = [a0.clamp(-1, 1)], self.t.sim_base(s, a0.clamp(-1, 1), grad=False)
        for k in range(1, self.t.K):
            u = self.t.cont_policy(y).clamp(-1, 1) if self.mode == 'policy' else self.a_ref[:, k]
            acts.append(u)
            if k < self.t.K - 1:
                y = self.t.sim_base(y, u, grad=False)
        return torch.stack(acts, 1)


class FirstActionTeacher(_GFTeacher):
    """REQUESTS R6: optimise the first action only; a continuation finishes the K-step plan.

    min_a ||w (G(x, a) - s*)||^2 + action_reg ||a - a_bar_0||^2 with the Stage I solver (lm_invert), G = one step
    with a then K - 1 steps of fa_cont (policy: the current policy closed loop; refs: a_bar_1..K-1 open loop),
    s* = the target of the best phase offset of the references' rollout. Jacobian fa_jacobian (fd: A + 1 rollouts of
    K steps per iteration; broyden: once, then rank-1 updates), iters iterations. Cost per label:
    K + iters (A + 2) K (fd) or K + (A + 1) K + iters K (broyden), plus the verification rollout K.
    """

    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        self.cont = str(cfg.get('fa_cont', 'policy'))
        if self.cont not in ('policy', 'refs'):
            raise ValueError(f'teacher.fa_cont must be policy or refs, not {self.cont!r}')
        self.fa_jac = str(cfg.get('fa_jacobian', 'fd'))
        self.fd_h = float(cfg.get('fd_h', 1e-2))
        self.cont_policy = None

    def anchor_to(self, policy):
        super().anchor_to(policy)
        self.cont_policy = policy

    @torch.no_grad()
    def optimise(self, sim, x, traj, j, a_ref):
        from .mentor import lm_invert

        if self.cont == 'policy' and self.cont_policy is None:
            raise ValueError('teacher.fa_cont=policy needs the policy (anchor_to)')
        tgt, _ = self.fixed_target(x, a_ref, traj, j)
        G = _Continuation(self, a_ref, self.cont)
        a0, _ = lm_invert(G, x, tgt, self.w, iters=self.iters, starts=1, h=self.fd_h, reg=self.lam, target=a_ref[:, 0],
                          x0=a_ref[:, 0], jacobian=self.fa_jac)
        plan = G.plan(x, a0)
        self.iters_used.append(self.iters)
        return plan, self.one_cost(x, plan, traj, j, a_ref)


class DiagnosticRejoin(RejoinTeacher):
    """DIAGNOSTIC ONLY (REQUESTS R3; uses simulator gradients, never a TUTOR arm): the gradient teacher's label, then
    label = u0_warm + diag_shrink (u0_adam - u0_warm) + diag_noise N(0, I), clipped to [-1, 1]. diag_shrink < 1 tests
    the "no correction" reading, diag_noise > 0 the "arbitrary plans" one. The gate reads the gradient plan's cost."""

    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        self.shrink = float(cfg.get('diag_shrink', 1.0))
        self.noise = float(cfg.get('diag_noise', 0.0))
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(cfg.get('gf_seed', 0)))
        if not self.screen:
            raise ValueError('teacher.kind=rejoin_diag needs screening (teacher.screen and num_windows > 1)')

    def optimise(self, sim, x, traj, j, a_ref):
        self._warm0 = a_ref[:, 0].detach().clone()
        return super().optimise(sim, x, traj, j, a_ref)

    def label(self, x):
        u, c = super().label(x)
        warm = getattr(self, '_warm0', u)
        out = warm + self.shrink * (u - warm)
        if self.noise > 0:
            out = out + self.noise * torch.randn(out.shape, device=out.device, generator=self.gen)
        return out.clamp(-1, 1), c


GF_TEACHERS = {'fdadam': FDAdamTeacher, 'lmplan': LMPlanTeacher, 'mppi': MPPITeacher, 'hindsight': HindsightTeacher,
               'firstact': FirstActionTeacher, 'rejoin_diag': DiagnosticRejoin}
