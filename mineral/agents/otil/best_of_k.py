from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Utils ----------
def pairwise_sqdist(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    XX = (X**2).sum(-1, keepdim=True)
    YY = (Y**2).sum(-1, keepdim=True).transpose(-2, -1)
    return (XX + YY - 2 * X @ Y.transpose(-2, -1)).clamp_min(0.0)

def pairwise_cosine_distance(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # X, Y: [BK, T, d]
    Xn = X / (X.norm(dim=-1, keepdim=True) + eps)
    Yn = Y / (Y.norm(dim=-1, keepdim=True) + eps)
    # sim: [BK, T, T]
    sim = torch.bmm(Xn, Yn.transpose(1, 2))
    # distance in [0, 2]
    return 1.0 - sim

def log_sinkhorn(C: torch.Tensor, eps: float = 0.1, iters: int = 60) -> torch.Tensor:
    B, T, _ = C.shape
    log_K = -C / eps
    log_u = torch.zeros(B, T, device=C.device, dtype=C.dtype)
    log_v = torch.zeros(B, T, device=C.device, dtype=C.dtype)
    log_a = torch.full(
        (B, T),
        -torch.log(torch.tensor(T, device=C.device, dtype=C.dtype)),
        device=C.device,
        dtype=C.dtype,
    )
    log_b = log_a.clone()

    lse = torch.logsumexp
    for _ in range(iters):
        log_u = log_a - lse(log_K + log_v.unsqueeze(1), dim=-1)
        log_v = log_b - lse(log_K.transpose(-2, -1) + log_u.unsqueeze(1), dim=-1)

    return (log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(-2)).exp()


class IdentityFeat(nn.Module):
    def forward(self, x):
        return x


class MLPFeat(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        y = self.net(x)
        return F.layer_norm(y, y.shape[-1:])


# ---------- Criterion base + implementations ----------
class BestOfKCriterion(nn.Module):
    """Given:
      sim_f: [B, T, d_feat]
      exp_f: [B, K, T, d_feat]

    Return:
      Either Lk: [B, K]   (per-crop costs)
      or (Lk, info_dict)
    """

    def forward(self, sim_f: torch.Tensor, exp_f: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OTSinkhornCriterion(BestOfKCriterion):
    def __init__(
        self,
        eps: float = 0.1,
        iters: int = 60,
        use_huber_speedup: bool = False,
        cost_type: Literal["huber", "l2", "cosine"] = "l2",
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.eps = eps
        self.iters = iters
        self.use_huber_speedup = use_huber_speedup
        self.cost_type = cost_type
        self.huber_delta = huber_delta

    def _cost_matrix(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        use_huber: bool | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        use_huber = use_huber or (self.cost_type == "huber")

        # X, Y: [BK, T, d]
        if use_huber:
            C2 = pairwise_sqdist(X, Y).clamp_min(1e-12)
            C = torch.sqrt(C2 + 1e-12)
            delta = self.huber_delta
            quad = 0.5 * C2
            lin = delta * (C - 0.5 * delta)
            return torch.where(C <= delta, quad, lin), C2
        elif self.cost_type == "cosine":
            C = pairwise_cosine_distance(X, Y)
            return C, None
        elif self.cost_type == "l2":
            return pairwise_sqdist(X, Y), None
        else:
            raise ValueError(f"Invalid cost_type: {self.cost_type}")

    def forward(self, sim_f: torch.Tensor, exp_f: torch.Tensor):
        B, K, T, d = exp_f.shape
        sim_f_tiled = sim_f.unsqueeze(1).expand(B, K, T, d).contiguous().view(B * K, T, d)
        exp_f_flat = exp_f.contiguous().view(B * K, T, d)

        if not self.cost_type == "huber" and self.use_huber_speedup:
            # use huber cost for sinkhorn speedup, but final cost is l2
            C_huber, C = self._cost_matrix(sim_f_tiled, exp_f_flat, use_huber=True)  # [B*K, T, T]
            P = log_sinkhorn(C_huber, eps=self.eps, iters=self.iters)  # [B*K, T, T]
        else:
            C, _ = self._cost_matrix(sim_f_tiled, exp_f_flat)  # [B*K, T, T]
            P = log_sinkhorn(C, eps=self.eps, iters=self.iters)  # [B*K, T, T]

        plan_costs = (P.detach() * C).sum(dim=-1).view(B, K, T)  # cost per sim timestep
        Lk = plan_costs.sum(dim=-1)  # [B, K]
        info = {"per_step_costs": plan_costs}
        return Lk, info


class SequenceRegressionCriterion(BestOfKCriterion):
    """Simple aligned (t-to-t) sequence MSE (or Huber) on features."""

    def __init__(self, use_huber: bool = False, huber_delta: float = 1.0, reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.reduction = reduction

    def forward(self, sim_f: torch.Tensor, exp_f: torch.Tensor) -> torch.Tensor:
        # sim_f: [B,T,d], exp_f: [B,K,T,d]
        B, K, T, d = exp_f.shape
        sim = sim_f.unsqueeze(1).expand(B, K, T, d)  # [B,K,T,d]
        diff = sim - exp_f

        if self.use_huber:
            # Smooth L1 over last dim (features), then average over time
            if self.reduction == "mean":
                per_t = F.smooth_l1_loss(exp_f, sim, beta=self.huber_delta, reduction="none")  # [B,K,T,d]
                Lk = per_t.mean(dim=(2, 3))  # mean over T,d
            else:
                per_t = F.smooth_l1_loss(exp_f, sim, beta=self.huber_delta, reduction="none")
                Lk = per_t.sum(dim=(2, 3))
        else:
            if self.reduction == "mean":
                Lk = (diff * diff).mean(dim=(2, 3))  # mean over T,d
            else:
                Lk = (diff * diff).sum(dim=(2, 3))
        # Keep per-step costs as mean over features for consistency
        per_step_costs = (diff * diff).mean(dim=3)
        return Lk, {"per_step_costs": per_step_costs}


class SequenceCosineCriterion(BestOfKCriterion):
    def forward(self, sim_f, exp_f):
        # normalize on feature dim
        sim_n = F.normalize(sim_f, dim=-1).unsqueeze(1)  # [B,1,T,d]
        exp_n = F.normalize(exp_f, dim=-1)  # [B,K,T,d]
        cos = (sim_n * exp_n).sum(-1)  # [B,K,T]
        # distance = 1 - cosine; mean over time
        Lk = (1.0 - cos).mean(dim=2)  # [B,K]
        per_step_costs = 1.0 - cos
        return Lk, {"per_step_costs": per_step_costs}


# ---------- Best-of-K wrapper that uses a criterion ----------
class BestOfK(nn.Module):
    def __init__(
        self,
        *,
        T: int,
        K: int = 8,
        tau: float = 0.5,
        use_mlp_features: bool = False,
        feature_dim: Optional[int] = None,
        embed_dim: int = 64,
        return_per_step_costs: bool = True,
        input_type: Literal["state", "state_state"] = "state",
        detach_prev_obs: bool = False,
        criterion: Optional[BestOfKCriterion] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.T = T
        self.K = K
        self.tau = tau
        self.input_type = input_type
        self.detach_prev_obs = detach_prev_obs
        self.use_mlp_features = use_mlp_features
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.return_per_step_costs = return_per_step_costs
        self.device = device

        if self.input_type not in ("state", "state_state"):
            raise ValueError(f"Invalid input_type: {self.input_type}")

        # Features
        if use_mlp_features:
            assert feature_dim is not None, "Set feature_dim when use_mlp_features=True"
            in_dim = feature_dim * 2 if self.input_type == "state_state" else feature_dim
            self.feat = MLPFeat(in_dim, embed_dim).to(device=device)
        else:
            self.feat = IdentityFeat()

        assert criterion is not None, "Must provide a criterion"
        self.criterion = criterion

    def _stack_consecutive_states(self, seq: torch.Tensor) -> torch.Tensor:
        """If configured, stack consecutive states along the feature dimension."""
        if self.input_type == "state":
            return seq, seq.shape[-2], seq.shape[-1]
        elif self.input_type == "state_state":
            obs = seq[..., :-1, :]  # [B, T-1, D]
            next_obs = seq[..., 1:, :]  # [B, T-1, D]
            if self.detach_prev_obs:
                obs = obs.detach()
            return torch.cat([obs, next_obs], dim=-1), obs.shape[-2], obs.shape[-1] * 2  # [B, T-1, 2D]
        else:
            raise ValueError(f"Invalid input_type: {self.input_type}")

    # ---- helpers (unchanged sampling logic) ----
    @torch.no_grad()
    def _sample_expert_ids_and_starts(self, B: int, K: int, expert_lens: torch.Tensor, T: int, device):
        valid = expert_lens >= T
        valid_ids = torch.where(valid)[0]
        if valid_ids.numel() == 0:
            raise ValueError(f"No expert trajectory has length >= T={T}")

        idx = torch.randint(0, valid_ids.numel(), (B, K), device=device)
        expert_ids = valid_ids[idx]  # [B, K]

        max_starts = (expert_lens[expert_ids] - T).clamp_min(0)  # [B, K]
        starts = torch.floor(torch.rand(B, K, device=device) * (max_starts + 1)).long()
        return expert_ids, starts  # [B,K], [B,K]

    def _gather_crops_from_bank(
        self, bank: torch.Tensor, expert_ids: torch.Tensor, starts: torch.Tensor, T: int
    ) -> torch.Tensor:
        B, K = expert_ids.shape
        _, Nmax, d = bank.shape
        t = torch.arange(T, device=bank.device).view(1, 1, T)  # [1,1,T]
        time_idx = starts.unsqueeze(-1) + t  # [B,K,T]
        exp_idx = expert_ids.unsqueeze(-1).expand(B, K, T)  # [B,K,T]
        crops = bank[exp_idx, time_idx, :]  # [B,K,T,d]
        return crops

    def forward(
        self,
        sim_seq: torch.Tensor,  # [B, Ts, d]
        expert,  # [B, N, d] OR bank [M, Nmax, d]
        *,
        expert_lens: Optional[torch.Tensor] = None,  # [M] if bank; ignored if [B,N,d]
        sim_is_window: bool = False,
        sim_start: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        device = sim_seq.device

        expert = expert.detach()

        # Make sim window
        if sim_is_window:
            assert sim_seq.shape[1] == self.T, f"sim_seq must be length T={self.T}"
            sim_win = sim_seq
        else:
            Ts = sim_seq.shape[1]
            assert Ts >= self.T, f"sim_seq length {Ts} must be >= T={self.T}"
            s0 = 0 if sim_start is None else int(sim_start)
            s0 = max(0, min(s0, Ts - self.T))
            sim_win = sim_seq[:, s0 : s0 + self.T, :]  # [B,T,d]
        B, T, d_in = sim_win.shape

        # Determine expert source
        if expert.dim() == 3 and expert.shape[0] != B:
            # expert bank mode [M,Nmax,d]
            bank = expert
            M, Nmax, d_e = bank.shape
            assert d_e == d_in, f"expert dim {d_e} != sim dim {d_in}"

            if expert_lens is None:
                expert_lens = torch.full((M,), Nmax, device=device, dtype=torch.long)
            else:
                assert expert_lens.shape == (M,), "expert_lens must be [M]"
                assert (expert_lens <= Nmax).all()

            expert_ids, starts = self._sample_expert_ids_and_starts(B, self.K, expert_lens, T, device)
            expert_crops = self._gather_crops_from_bank(bank, expert_ids, starts, T)  # [B,K,T,d]
        elif expert.dim() == 3 and expert.shape[0] == B:
            # per-batch experts [B,N,d]
            N = expert.shape[1]

            starts = torch.randint(0, max(N - T, 0) + 1, (B, self.K), device=device)
            idx_t = torch.arange(T, device=device).view(1, 1, T)
            idx = starts.unsqueeze(-1) + idx_t
            b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, self.K, T)
            expert_crops = expert[b_idx, idx, :]  # [B,K,T,d]
            expert_ids = None
        else:
            raise ValueError("expert must be either [B,N,d] or bank [M,Nmax,d]")

        # ---- Features ----
        sim_win, T, d_in = self._stack_consecutive_states(sim_win)
        expert_crops, _, d_e = self._stack_consecutive_states(expert_crops)
        sim_f = self.feat(sim_win)  # [B,T,d’]
        exp_f = self.feat(expert_crops.view(B * self.K, T, -1))  # [B*K,T,d’]
        exp_f = exp_f.view(B, self.K, T, -1)  # [B,K,T,d’]

        # ---- Per-crop costs via pluggable criterion ----
        crit_out = self.criterion(sim_f, exp_f)
        if isinstance(crit_out, tuple):
            Lk, crit_info = crit_out
        else:
            Lk, crit_info = crit_out, {}

        # ---- Softmin over K ----
        loss = -self.tau * torch.logsumexp(-Lk / self.tau, dim=1).mean()
        weighted_per_step_costs = None
        if self.return_per_step_costs and "per_step_costs" in crit_info:
            per_step_costs = crit_info["per_step_costs"]  # [B,K,T]
            weights = torch.softmax(-Lk / self.tau, dim=1).unsqueeze(-1)
            weighted_per_step_costs = (weights * per_step_costs).sum(dim=1)  # [B,T]

        info = {
            "Lk": Lk.detach(),
            "Lk_min": Lk.detach().min(dim=1).values.mean(),
            "Lk_mean": Lk.detach().mean(dim=1).mean(),
            "starts": starts.detach(),
            "best_idx": Lk.argmin(dim=1).detach(),
        }
        if weighted_per_step_costs is not None and self.return_per_step_costs:
            info["per_step_costs"] = weighted_per_step_costs
        if expert.dim() == 3 and expert.shape[0] != B:
            info["expert_ids"] = expert_ids.detach() if expert_ids is not None else None

        return loss, info


# ---------- Example usage ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B_env, Ts = 100, 64  # 100 parallel envs
    M_expert, N = 10, 1000
    d_in, T = 48, 32

    sim_seq = torch.randn(B_env, Ts, d_in, device=device)
    expert_bank = torch.randn(M_expert, N, d_in, device=device)
    expert_lens = torch.randint(N - 50, N + 1, (M_expert,), device=device)

    # --- OT Sinkhorn (default) ---
    ot_crit = OTSinkhornCriterion(eps=0.1, iters=60, use_huber=False, huber_delta=1.0)
    loss_fn_ot = BestOfK(
        T=T,
        K=8,
        tau=0.5,
        use_mlp_features=True,
        feature_dim=d_in,
        embed_dim=64,
        return_per_step_costs=True,
        criterion=ot_crit,
    ).to(device)
    loss_ot, info_ot = loss_fn_ot(sim_seq, expert_bank, expert_lens=expert_lens, sim_is_window=False)
    print("OT Loss:", loss_ot.item(), "Best (first 5):", info_ot["best_idx"][:5])

    # --- MSE (aligned) ---
    mse_crit = SequenceRegressionCriterion(use_huber=False, reduction="mean")
    loss_fn_mse = BestOfK(
        T=T,
        K=8,
        tau=0.5,
        use_mlp_features=True,
        feature_dim=d_in,
        embed_dim=64,
        return_per_step_costs=True,
        criterion=mse_crit,
    ).to(device)
    loss_mse, info_mse = loss_fn_mse(sim_seq, expert_bank, expert_lens=expert_lens, sim_is_window=False)
    print("MSE Loss:", loss_mse.item(), "Best (first 5):", info_mse["best_idx"][:5])

    # --- Cosine (aligned) ---
    cos_crit = SequenceCosineCriterion()
    loss_fn_cos = BestOfK(
        T=T,
        K=8,
        tau=0.5,
        use_mlp_features=True,
        feature_dim=d_in,
        embed_dim=64,
        return_per_step_costs=True,
        criterion=cos_crit,
    ).to(device)
    loss_cos, info_cos = loss_fn_cos(sim_seq, expert_bank, expert_lens=expert_lens, sim_is_window=False)
    print("Cosine Loss:", loss_cos.item(), "Best (first 5):", info_cos["best_idx"][:5])
