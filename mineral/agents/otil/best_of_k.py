from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_sqdist(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    XX = (X**2).sum(-1, keepdim=True)
    YY = (Y**2).sum(-1, keepdim=True).transpose(-2, -1)
    return (XX + YY - 2 * X @ Y.transpose(-2, -1)).clamp_min(0.0)


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

    def lse(x, dim=-1):
        return torch.logsumexp(x, dim=dim)

    for _ in range(iters):
        log_u = log_a - lse(log_K + log_v.unsqueeze(1), dim=-1)
        log_v = log_b - lse(log_K.transpose(-2, -1) + log_u.unsqueeze(1), dim=-1)

    return (log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(-2)).exp()


class IdentityFeat(nn.Module):
    def __init__(self):
        super().__init__()

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


# --- Best-of-K that supports expert bank [M, N, d] ---
@dataclass
class BestOfKConfig:
    T: int
    K: int = 8
    eps: float = 0.1
    sinkhorn_iters: int = 60
    tau: float = 0.5
    use_mlp_features: bool = False
    feature_dim: Optional[int] = None
    embed_dim: int = 64
    use_huber: bool = False
    huber_delta: float = 1.0
    action_weight: float = 1.0


class BestOfKSoftminOT(nn.Module):
    def __init__(self, cfg: BestOfKConfig, device=None, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        if cfg.use_mlp_features:
            assert cfg.feature_dim is not None, "Set feature_dim when use_mlp_features=True"
            self.feat = MLPFeat(cfg.feature_dim, cfg.embed_dim)
        else:
            self.feat = IdentityFeat()

    def _cost_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_huber:
            C2 = pairwise_sqdist(X, Y).clamp_min(1e-12)
            C = torch.sqrt(C2 + 1e-12)
            delta = self.cfg.huber_delta
            quad = 0.5 * C2
            lin = delta * (C - 0.5 * delta)
            return torch.where(C <= delta, quad, lin)
        else:
            return pairwise_sqdist(X, Y)

    @torch.no_grad()
    def _sample_expert_ids_and_starts(
        self,
        B: int,
        K: int,
        expert_lens: torch.Tensor,
        T: int,
        device,  # [M]
    ):
        # Only choose experts with length >= T
        valid = expert_lens >= T
        valid_ids = torch.where(valid)[0]
        if valid_ids.numel() == 0:
            raise ValueError(f"No expert trajectory has length >= T={T}")

        # Sample expert id per (b,k)
        idx = torch.randint(0, valid_ids.numel(), (B, K), device=device)
        expert_ids = valid_ids[idx]  # [B, K]

        # Sample start per (b,k) within that expert's valid range
        max_starts = (expert_lens[expert_ids] - T).clamp_min(0)  # [B, K]
        # Uniform integer in [0, max_start]
        starts = torch.floor(torch.rand(B, K, device=device) * (max_starts + 1)).long()
        return expert_ids, starts  # [B,K], [B,K]

    def _gather_crops_from_bank(
        self,
        bank: torch.Tensor,  # [M, Nmax, d]
        expert_ids: torch.Tensor,  # [B, K]
        starts: torch.Tensor,  # [B, K]
        T: int,
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
        expert,  # either [B, N, d] OR bank [M, Nmax, d]
        *,
        expert_lens: Optional[torch.Tensor] = None,  # [M] if bank; ignored if [B,N,d]
        sim_is_window: bool = False,
        sim_start: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        cfg = self.cfg
        device = sim_seq.device

        expert = expert.detach()

        # Make sim window
        if sim_is_window:
            assert sim_seq.shape[1] == cfg.T, f"sim_seq must be length T={cfg.T}"
            sim_win = sim_seq
        else:
            Ts = sim_seq.shape[1]
            assert Ts >= cfg.T, f"sim_seq length {Ts} must be >= T={cfg.T}"
            s0 = 0 if sim_start is None else int(sim_start)
            s0 = max(0, min(s0, Ts - cfg.T))
            sim_win = sim_seq[:, s0 : s0 + cfg.T, :]  # [B,T,d]
        B, T, d_in = sim_win.shape

        # Optional action weighting (assume last half are actions)
        if getattr(cfg, "action_weight", 1.0) != 1.0:
            split = d_in // 2

            def weight(sa):
                s, a = sa[..., :split], sa[..., split:]
                a = a * cfg.action_weight
                return torch.cat([s, a], dim=-1)

            sim_win = weight(sim_win)

        # Determine expert source mode
        if expert.dim() == 3 and expert.shape[0] != B:
            # Treat as bank [M, Nmax, d]
            bank = expert
            M, Nmax, d_e = bank.shape
            assert d_e == d_in, f"expert dim {d_e} != sim dim {d_in}"

            if expert_lens is None:
                # assume all same length Nmax
                expert_lens = torch.full((M,), Nmax, device=device, dtype=torch.long)
            else:
                assert expert_lens.shape == (M,), "expert_lens must be [M]"
                assert (expert_lens <= Nmax).all()

            # Sample K windows from the bank for each sim env
            expert_ids, starts = self._sample_expert_ids_and_starts(B, cfg.K, expert_lens, T, device)
            expert_crops = self._gather_crops_from_bank(bank, expert_ids, starts, T)  # [B,K,T,d]
        elif expert.dim() == 3 and expert.shape[0] == B:
            # Per-batch expert trajectories [B,N,d] (old behavior)
            N = expert.shape[1]
            if getattr(cfg, "action_weight", 1.0) != 1.0:
                expert = weight(expert)
            starts = torch.randint(0, max(N - T, 0) + 1, (B, cfg.K), device=device)
            # Gather
            idx_t = torch.arange(T, device=device).view(1, 1, T)
            idx = starts.unsqueeze(-1) + idx_t
            b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, cfg.K, T)
            expert_crops = expert[b_idx, idx, :]  # [B,K,T,d]
            expert_ids = None
        else:
            raise ValueError("expert must be either [B,N,d] or bank [M,Nmax,d]")

        # Embed features
        sim_f = self.feat(sim_win)  # [B,T,d’]
        if self.cfg.use_mlp_features:
            _d_feat = sim_f.shape[-1]
        # Flatten K dimension to compute OT per-crop
        exp_f = self.feat(expert_crops.view(B * cfg.K, T, -1)).view(B, cfg.K, T, -1)

        sim_f_tiled = sim_f.unsqueeze(1).expand(B, cfg.K, T, sim_f.shape[-1]).contiguous().view(B * cfg.K, T, -1)
        exp_f_flat = exp_f.contiguous().view(B * cfg.K, T, -1)

        C = self._cost_matrix(sim_f_tiled, exp_f_flat)  # [B*K, T, T]
        P = log_sinkhorn(C, eps=cfg.eps, iters=cfg.sinkhorn_iters).detach()  # [B*K, T, T]
        Lk = (P * C).sum(dim=(-1, -2)).view(B, cfg.K)  # [B, K]

        # Softmin over K
        tau = cfg.tau
        loss = -tau * torch.logsumexp(-Lk / tau, dim=1).mean()

        info = {
            "Lk": Lk.detach(),  # per-crop costs
            "Lk_min": Lk.detach().min(dim=1).values.mean(),
            "Lk_mean": Lk.detach().mean(dim=1).mean(),
            "starts": starts.detach(),  # chosen crop starts
            "best_idx": Lk.argmin(dim=1).detach(),  # index of the best crop per batch
        }
        if expert.dim() == 3 and expert.shape[0] != B:
            info["expert_ids"] = expert_ids.detach() if expert_ids is not None else None

        return loss, info


# --- Example usage with B != M ---
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B_env, Ts = 100, 64  # 100 parallel envs
    M_expert, N = 10, 1000  # 10 expert trajectories
    d_in, T = 48, 32

    sim_seq = torch.randn(B_env, Ts, d_in, device=device)
    expert_bank = torch.randn(M_expert, N, d_in, device=device)
    expert_lens = torch.randint(N - 50, N + 1, (M_expert,), device=device)  # variable lengths near N

    cfg = BestOfKConfig(
        T=T,
        K=8,
        eps=0.1,
        sinkhorn_iters=60,
        tau=0.5,
        use_mlp_features=True,
        feature_dim=d_in,
        embed_dim=64,
    )
    loss_fn = BestOfKSoftminOT(cfg, device=device).to(device)

    loss, info = loss_fn(sim_seq, expert_bank, expert_lens=expert_lens, sim_is_window=False)
    print("Loss:", loss.item())
    print("Best crop per env (first 5):", info["best_idx"][:5])
    if "expert_ids" in info and info["expert_ids"] is not None:
        print("Expert ids for those crops (first env):", info["expert_ids"][0])
