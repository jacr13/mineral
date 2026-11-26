"""TUTOR: hindsight inverse teacher (HIT). Every simulated segment is an exact demonstration of reaching where it landed.

Data (all already simulated; nothing extra is rolled for it):
  - the demonstration windows with the Stage I actions: (s_j, a_hat_j..j+K-1, s'_{j+K-1}) for every admissible j;
  - the learner's rollouts of each round (MENTOR.collect, handed over by observe_rollout): every K-step window without
    a reset or a fall;
  - the teacher's own rollouts (the references, each corrected plan, the exploration plans).
Model: a goal-conditioned inverse model g(x0, xK) -> plan (K x A), trained on the buffer; hit_head=mse (regression,
the conditional mean plan) or flow (conditional flow matching on the plan, sampled from a fixed latent, so that the
same state and goal always give the same plan).
Label at a learner state x (windows, screening, anchored references as the rejoin teacher; target s* = the demo state
of the best phase offset of the references' rollout):
  p_0 = g(x, s*) (the references while the model is untrained); then hit_iters times: roll p, it lands at y,
  p <- clip(p + g(x, s*) - g(x, y)) (the model's error at y is known exactly, since p reaches y: the difference cancels
  its local bias); the plan with the lowest rejoin cost among the references and the iterates is the label.
  hit_explore noisy copies of the plan (radius sample_noise) are rolled for the buffer only.
Cost per label (forward steps): K (references) + hit_iters K + hit_explore K, plus screening and anchor.
"""

import torch
import torch.nn as nn

from .mentor_gf import _GFTeacher


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


class HindsightInverseTeacher(_GFTeacher):
    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        self.head = str(cfg.get('hit_head', 'mse'))
        if self.head not in ('mse', 'flow'):
            raise ValueError(f'teacher.hit_head must be mse or flow, not {self.head!r}')
        self.hit_iters = int(cfg.get('hit_iters', 3))
        self.explore = int(cfg.get('hit_explore', 4))
        self.sigma = float(cfg.get('sample_noise', 0.1))
        self.cap = int(cfg.get('hit_buffer', 1_000_000))
        self.round_steps = int(cfg.get('hit_round_steps', 2000))
        self.batch_steps = int(cfg.get('hit_batch_steps', 50))
        self.flow_steps = int(cfg.get('hit_flow_steps', 10))
        self.min_data = int(cfg.get('hit_min_data', 5000))
        D, KA = self.D, self.K * self.A
        valid = torch.arange(self.T, device=device)[None] < lengths[:, None]
        self.mu, self.sd = S[valid].mean(0), S[valid].std(0).clamp_min(1e-6)
        cond = 2 * D
        if self.head == 'mse':
            self.net = _MLP(cond, KA).to(device)
        else:  # velocity field v(z_t, t | x0, xK)
            self.net = _MLP(cond + KA + 1, KA).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(cfg.get('hit_lr', 1e-3)))
        self.buf = None  # (x0, xK, plan)
        self.trained = False
        self.stats = []
        # Stage I: the demonstration windows with the recovered actions are K-step segments
        n, T = S.shape[:2]
        jj = torch.arange(T - self.K, device=device)
        ok = (jj[None] + self.K) <= lengths[:, None]
        tr, j0 = ok.nonzero(as_tuple=True)
        idx = j0[:, None] + torch.arange(self.K, device=device)[None]
        self.add(S[tr, j0], S1[tr, j0 + self.K - 1], A_hat[tr[:, None], idx])

    # ---- data ----
    @torch.no_grad()
    def add(self, x0, xK, plan):
        ok = torch.isfinite(x0).all(-1) & torch.isfinite(xK).all(-1) & torch.isfinite(plan).flatten(1).all(-1)
        new = (x0[ok], xK[ok], plan[ok].reshape(-1, self.K * self.A).clamp(-1, 1))
        if self.buf is None:
            self.buf = new
        else:
            self.buf = tuple(torch.cat([b, c])[-self.cap :] for b, c in zip(self.buf, new))

    @torch.no_grad()
    def observe_rollout(self, roll):
        obs, act, brk = roll['obs'], roll['act'], roll['broken']  # [T, n, D], [T, n, A], [T, n]
        T = obs.shape[0]
        K = self.K
        xs, ys, ps = [], [], []
        for t in range(T - K):
            ok = ~brk[t : t + K + 1].any(0)  # no fall in t..t+K-1 and no reset up to t+K (obs[t+K] must follow act[t+K-1])
            if ok.any():
                xs.append(obs[t][ok])
                ys.append(obs[t + K][ok])
                ps.append(act[t : t + K].transpose(0, 1)[ok])
        if xs:
            self.add(torch.cat(xs), torch.cat(ys), torch.cat(ps))
        self.fit(self.round_steps)

    # ---- model ----
    def cond(self, x0, xK):
        return torch.cat([(x0 - self.mu) / self.sd, (xK - x0) * self.w], -1)

    def loss(self, x0, xK, plan):
        c = self.cond(x0, xK)
        if self.head == 'mse':
            return ((torch.tanh(self.net(c)) - plan) ** 2).mean()
        z0 = torch.randn_like(plan)
        t = torch.rand(plan.shape[0], 1, device=plan.device)
        zt = (1 - t) * z0 + t * plan
        return ((self.net(torch.cat([c, zt, t], -1)) - (plan - z0)) ** 2).mean()

    def fit(self, steps, bs=1024):
        if self.buf is None or self.buf[0].shape[0] < self.min_data or steps <= 0:
            return
        N = self.buf[0].shape[0]
        self.net.train()
        with torch.enable_grad():
            for _ in range(steps):
                i = torch.randint(0, N, (bs,), device=self.device, generator=self.gen)
                loss = self.loss(self.buf[0][i], self.buf[1][i], self.buf[2][i])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        self.net.eval()
        self.trained = True
        self.last_loss = loss.item()

    @torch.no_grad()
    def predict(self, x0, xK):
        c = self.cond(x0, xK)
        if self.head == 'mse':
            out = torch.tanh(self.net(c))
        else:  # Euler integration of the flow from a fixed latent (zero): one deterministic plan per (x0, xK)
            z = torch.zeros(x0.shape[0], self.K * self.A, device=x0.device)
            h = 1.0 / self.flow_steps
            for k in range(self.flow_steps):
                t = torch.full((x0.shape[0], 1), k * h, device=x0.device)
                z = z + h * self.net(torch.cat([c, z, t], -1))
            out = z.clamp(-1, 1)
        return out.view(-1, self.K, self.A)

    # ---- label ----
    @torch.no_grad()
    def optimise(self, sim, x, traj, j, a_ref):
        n = x.shape[0]
        tgt, e0 = self.fixed_target(x, a_ref, traj, j)
        y_ref, out_ref = self._ref_rollout  # the references' rollout done by fixed_target
        self.add(x, y_ref, a_ref)
        best_a, best_c = a_ref.clone(), self.cost((y_ref, out_ref), a_ref, traj, j, a_ref, x)
        st = {'trained': float(self.trained)}
        if self.trained:
            g_star = self.predict(x, tgt)
            p = g_star.clone()
            for it in range(self.hit_iters):
                y, out = self.rollout(self.sim_base, x, p, grad=False)
                self.add(x, y, p)
                c = self.cost((y, out), p, traj, j, a_ref, x)
                better = c < best_c
                best_a = torch.where(better[:, None, None], p, best_a)
                best_c = torch.where(better, c, best_c)
                if it == 0:
                    st['model_err_ratio'] = (((y - tgt) * self.w).norm(dim=-1) / e0.norm(dim=-1).clamp_min(1e-9)).median().item()
                p = (p + g_star - self.predict(x, y)).clamp(-1, 1)
            st['best_is_ref'] = (best_a == a_ref).all(-1).all(-1).float().mean().item()
        for _ in range(self.explore):  # buffer only
            q = (best_a + self.sigma * torch.randn(best_a.shape, device=x.device, generator=self.gen)).clamp(-1, 1)
            y, _ = self.rollout(self.sim_base, x, q, grad=False)
            self.add(x, y, q)
        self.fit(self.batch_steps)
        self.stats.append(st)
        self.iters_used.append(self.hit_iters)
        return best_a, best_c
