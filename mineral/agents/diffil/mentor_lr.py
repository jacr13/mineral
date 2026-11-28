"""TUTOR: Levenberg-Marquardt on a low-rank K-step plan (teacher.kind=lmlr), forward steps only.

The plan is a = clip(a_bar + B z): B (K A x r) is an orthonormal basis, the product of lr_time cosine functions over the
K plan steps (phi_k(t) = cos(pi k (t + 1/2) / K), k < lr_time; smooth corrections first) and an action basis (identity,
or lr_act > 0: the top principal directions of the Stage I actions, for over-actuated bodies). LM runs on z exactly as
lmplan runs on the plan (residual [w (x_K - s*), sqrt(action_reg) (a - a_ref)], best phase offset re-chosen each
iteration): the forward-difference Jacobian costs r + 1 rollouts instead of K A + 1, then Broyden rank-1 updates
(lm_jacobian=broyden) or a new forward-difference Jacobian every iteration (fd).
Cost per label: (r + 1) K + iters K (broyden), plus screening num_windows K and anchor K - 1, and K for the final cost.
"""

import math

import torch

from .mentor_gf import LMPlanTeacher


class LowRankLMTeacher(LMPlanTeacher):
    def __init__(self, spec, S, S1, A_hat, lengths, batch, device, counter, cfg):
        super().__init__(spec, S, S1, A_hat, lengths, batch, device, counter, cfg)
        K, A = self.K, self.A
        n_t = max(1, min(int(cfg.get('lr_time', 3)), K))
        t = torch.arange(K, device=device, dtype=torch.float32)
        phi = torch.stack([torch.cos(math.pi * k * (t + 0.5) / K) for k in range(n_t)], 1)  # (K, n_t)
        phi = phi / phi.norm(dim=0, keepdim=True)
        n_a = int(cfg.get('lr_act', 0))
        if n_a > 0 and n_a < A:  # principal directions of the Stage I actions (valid demo steps)
            valid = torch.arange(A_hat.shape[1], device=device)[None] < lengths[:, None]
            acts = A_hat[valid].float()
            U = torch.linalg.svd(acts - acts.mean(0), full_matrices=False).Vh[:n_a].T  # (A, n_a)
        else:
            U = torch.eye(A, device=device)
        # B[(t, i), (k, j)] = phi[t, k] U[i, j]  (plan flattened time-major, as everywhere: index t * A + i)
        self.B = torch.einsum('tk,ij->tikj', phi, U).reshape(K * A, -1).contiguous()  # orthonormal columns
        self.r = self.B.shape[1]

    @torch.no_grad()
    def plan_of(self, ar, z):
        return (ar + z @ self.B.T).clamp(-1, 1)

    @torch.no_grad()
    def jacobian_z(self, x, ar, z, y):
        n, r, h = x.shape[0], self.r, self.fd_h
        zp = z[:, None] + h * torch.eye(r, device=x.device)[None]  # (n, r, r)
        plans = (ar[:, None] + zp @ self.B.T).clamp(-1, 1)  # (n, r, K A)
        yp = self.final_states(x, plans.view(n, r, self.K, self.A))  # (n, r, D)
        return ((yp - y[:, None]) / h).transpose(1, 2) * self.w[None, :, None]  # (n, D, r)

    @torch.no_grad()
    def optimise(self, sim, x, traj, j, a_ref):
        n, KA, r = x.shape[0], self.K * self.A, self.r
        ar = a_ref.reshape(n, KA)
        z = torch.zeros(n, r, device=x.device)
        flat = self.plan_of(ar, z)
        y, rs, ra, cost = self.residual(x, flat, traj, j, ar)
        lam = torch.full_like(cost, self.lam0)
        J = None
        I = torch.eye(r, device=x.device)[None]
        Bt = self.B.T  # (r, K A)
        for it in range(self.iters):
            if J is None or self.jac == 'fd':
                J = self.jacobian_z(x, ar, z, y)
            H = J.transpose(1, 2) @ J + self.lam * I  # B orthonormal: d(r_act)/dz = sqrt(lam) B
            g = (J.transpose(1, 2) @ rs[..., None]).squeeze(-1) + (self.lam**0.5) * (ra @ Bt.T)
            A_ = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2).clamp_min(1e-12))
            z_new = z - torch.linalg.solve(A_, g[..., None]).squeeze(-1)
            new = self.plan_of(ar, z_new)
            y_n, rs_n, ra_n, c_n = self.residual(x, new, traj, j, ar)
            if self.jac == 'broyden':
                dz = z_new - z
                dy = (y_n - y) * self.w
                den = (dz**2).sum(-1, keepdim=True).clamp_min(1e-12)
                J = J + ((dy - (J @ dz[..., None]).squeeze(-1)) / den)[..., None] * dz[:, None]
            ok = c_n < cost
            z = torch.where(ok[:, None], z_new, z)
            flat = torch.where(ok[:, None], new, flat)
            y = torch.where(ok[:, None], y_n, y)
            rs = torch.where(ok[:, None], rs_n, rs)
            ra = torch.where(ok[:, None], ra_n, ra)
            cost = torch.where(ok, c_n, cost)
            lam = torch.where(ok, (lam * 0.3).clamp_min(1e-9), (lam * 5.0).clamp_max(1e9))
        self.iters_used.append(self.iters)
        a = flat.view(n, self.K, self.A)
        return a, self.one_cost(x, a, traj, j, a_ref)
