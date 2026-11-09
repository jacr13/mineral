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


def log_sinkhorn_unbalanced(
    C: torch.Tensor,
    eps: float = 0.1,
    iters: int = 60,
    rho: float = 1.0,
) -> torch.Tensor:
    """Entropic OT with *relaxed* marginals (unbalanced Sinkhorn).

    Reviewer Q9cS: "Some transport plans in Fig. 3(c) and Fig. 9(f)(h) show
    fragmented patterns, potentially caused by the uniform marginal constraints
    of OT". Balanced OT must move every unit of agent mass somewhere, so an
    agent state with no good expert match is still forced to couple with
    something -- which is what produces the fragmented rows.

    Unbalanced OT replaces the hard marginal constraints with KL penalties of
    strength ``rho``, letting the plan leave mass unmatched when nothing fits.
    The standard scaling form multiplies each Sinkhorn update by
    ``rho / (rho + eps)`` (Chizat et al., 2018; Sejourne et al., 2019);
    ``rho -> inf`` recovers the balanced iteration exactly, which is asserted
    in the tests.

    Args:
        C: ``[B, T, L]`` cost matrix.
        eps: entropic regularisation.
        iters: Sinkhorn iterations.
        rho: marginal relaxation. Smaller = more mass may go unmatched.
    """
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
    scale = rho / (rho + eps)

    lse = torch.logsumexp
    for _ in range(iters):
        log_u = scale * (log_a - lse(log_K + log_v.unsqueeze(1), dim=-1))
        log_v = scale * (log_b - lse(log_K.transpose(-2, -1) + log_u.unsqueeze(1), dim=-1))

    log_P = log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(-2)

    # Relaxing the marginals also frees the plan's total mass: it rises from
    # 1.0 (balanced) to ~15 at rho=0.01 on random costs. Downstream the per-step
    # cost is <C, mu>, so that would rescale the reward by the same factor and
    # the comparison against balanced OT would be measuring scale rather than
    # the marginal relaxation -- the same confound found between the L2 and
    # cosine ground costs. Renormalising the *total* mass to 1 fixes the scale
    # while leaving the relative row masses free, which is the whole point:
    # an agent state with no good expert match can still carry less mass than
    # its neighbours.
    # Normalise in the LOG domain, before exponentiating. Doing it afterwards
    # is not equivalent: scale = rho/(rho+eps) < 1 damps the Sinkhorn updates,
    # so log_u + log_v no longer offsets log_K (which reaches -934 at eps=0.1
    # on these costs) and exp() underflows -- the raw plan mass came out at
    # 3.5e-13 for some batch elements, below the clamp that was used here, so
    # dividing by the clamped sum returned P * 1e12 and left those elements
    # with a total mass of 0.34-0.36 instead of 1. That silently rescaled the
    # per-step costs for part of the batch and invalidated the first
    # unbalanced-OT runs (9 Nov 2026).
    log_P = log_P - torch.logsumexp(log_P.flatten(-2), dim=-1)[..., None, None]
    return log_P.exp()


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


def ground_cost_matrix(
    X: torch.Tensor,
    Y: torch.Tensor,
    cost_type: Literal["huber", "l2", "cosine"] = "l2",
    huber_delta: float = 1.0,
    force_huber: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pairwise ground cost between two feature sequences.

    Shared by every criterion. This is a plain distance computation -- it has
    nothing to do with Sinkhorn or with optimal transport, and lives outside
    OTSinkhornCriterion so that the soft-DTW path visibly does not depend on
    the OT path.

    Args:
        X, Y: ``[B, T, d]`` feature sequences.
        cost_type: "l2" (squared Euclidean), "cosine" (1 - cos), or "huber".
        huber_delta: transition point of the Huber cost.
        force_huber: use the Huber cost regardless of ``cost_type``; returns the
            squared-L2 matrix as the second element so a caller can align on one
            cost and score with the other.

    Returns:
        ``(C, C_sq_l2_or_None)``.
    """
    if force_huber or cost_type == "huber":
        C2 = pairwise_sqdist(X, Y).clamp_min(1e-12)
        C = torch.sqrt(C2 + 1e-12)
        quad = 0.5 * C2
        lin = huber_delta * (C - 0.5 * huber_delta)
        return torch.where(C <= huber_delta, quad, lin), C2
    if cost_type == "cosine":
        return pairwise_cosine_distance(X, Y), None
    if cost_type == "l2":
        return pairwise_sqdist(X, Y), None
    raise ValueError(f"Invalid cost_type: {cost_type}")


class OTSinkhornCriterion(BestOfKCriterion):
    def __init__(
        self,
        eps: float = 0.1,
        iters: int = 60,
        use_huber_speedup: bool = False,
        cost_type: Literal["huber", "l2", "cosine"] = "l2",
        huber_delta: float = 1.0,
        detach_plan: bool = True,
        marginal_rho: Optional[float] = None,
    ):
        super().__init__()
        self.eps = eps
        self.iters = iters
        self.use_huber_speedup = use_huber_speedup
        self.cost_type = cost_type
        self.huber_delta = huber_delta
        # detach_plan=True (the default, and what every reported run did) gives a
        # fixed-coupling surrogate gradient: the actor gradient flows through the
        # cost matrix only, at a frozen coupling. Setting it False differentiates
        # through the Sinkhorn iterations as well, which is what the paper's text
        # describes. Kept configurable so the two can be compared directly.
        self.detach_plan = detach_plan
        # None = balanced OT (uniform marginals, as published). A float relaxes
        # the marginals so unmatched agent states need not be coupled.
        self.marginal_rho = marginal_rho

    def _cost_matrix(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        use_huber: bool | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return ground_cost_matrix(
            X, Y,
            cost_type=self.cost_type,
            huber_delta=self.huber_delta,
            force_huber=bool(use_huber),
        )

    def _plan(self, C):
        if self.marginal_rho is None:
            return log_sinkhorn(C, eps=self.eps, iters=self.iters)
        return log_sinkhorn_unbalanced(C, eps=self.eps, iters=self.iters, rho=self.marginal_rho)

    def forward(self, sim_f: torch.Tensor, exp_f: torch.Tensor):
        B, K, T, d = exp_f.shape
        sim_f_tiled = sim_f.unsqueeze(1).expand(B, K, T, d).contiguous().view(B * K, T, d)
        exp_f_flat = exp_f.contiguous().view(B * K, T, d)

        if not self.cost_type == "huber" and self.use_huber_speedup:
            # use huber cost for sinkhorn speedup, but final cost is l2
            C_huber, C = self._cost_matrix(sim_f_tiled, exp_f_flat, use_huber=True)  # [B*K, T, T]
            P = self._plan(C_huber)  # [B*K, T, T]
        else:
            C, _ = self._cost_matrix(sim_f_tiled, exp_f_flat)  # [B*K, T, T]
            P = self._plan(C)  # [B*K, T, T]

        P_eff = P.detach() if self.detach_plan else P
        plan_costs = (P_eff * C).sum(dim=-1).view(B, K, T)  # cost per sim timestep
        Lk = plan_costs.sum(dim=-1)  # [B, K]
        info = {
            "per_step_costs": plan_costs,
            "transport_plans": P.detach().view(B, K, T, T),
        }
        return Lk, info


def soft_dtw(
    C: torch.Tensor,
    gamma: float = 0.1,
    open_end: bool = False,
    subsequence: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched soft-DTW over a cost matrix, differentiable in ``C``.

    Reviewers asked for an ordered-alignment comparison (soft-DTW / GDTW-family)
    against the Sinkhorn coupling, in an otherwise identical pipeline. This is a
    drop-in alternative to ``log_sinkhorn``: same signature shape, same return
    convention (a per-(agent, expert) weight matrix plus the scalar alignment
    value), so ``OTSinkhornCriterion`` and ``SoftDTWCriterion`` differ only in how
    the matching weights are produced.

    The recursion is the standard one with a soft minimum of temperature
    ``gamma``:

        R[i, j] = C[i, j] + softmin_gamma(R[i-1, j], R[i, j-1], R[i-1, j-1])

    ``subsequence`` frees the expert axis at both ends (row 0 of ``R`` costs
    nothing to enter, so the agent window may match any contiguous expert
    sub-segment) -- this is the "subsequence soft-DTW" variant one reviewer
    argued best-of-K approximates. ``open_end`` additionally frees the end of
    the expert axis, which is the open-end DTW another reviewer suggested as a
    way to avoid the fragmented plans that OT's uniform marginals can produce.

    Args:
        C: ``[B, T, L]`` cost matrix, agent timesteps on dim 1.
        gamma: soft-minimum temperature. ``gamma -> 0`` recovers hard DTW.
        open_end: do not force the alignment to finish at the last expert step.
        subsequence: do not force the alignment to start at the first expert
            step (and, with ``open_end``, allows any contiguous sub-segment).

    Returns:
        ``(value, W)`` where ``value`` is ``[B]``, the soft-DTW alignment cost,
        and ``W`` is ``[B, T, L]``, the gradient ``dvalue/dC``. ``W`` is the
        soft alignment matrix: it is non-negative, sums to the expected path
        length, and plays the role the transport plan plays for OT, so the
        per-step costs downstream are computed identically.
    """
    B, T, L = C.shape
    device = C.device
    big = 1e9

    # Anti-diagonal (wavefront) evaluation. In skewed coordinates s = i + j, the
    # cell (i, j) depends only on columns s-1 and s-2, and all cells sharing an s
    # are independent -- so the T*L sequential Python steps of the textbook
    # recursion collapse to T+L vectorized ones. With H = L = 32 that is ~65
    # iterations instead of ~1024, which matters because this runs inside every
    # actor update. (Measured: the naive loop cost ~45% of epoch time.)
    i_idx = torch.arange(1, T + 1, device=device)  # agent rows 1..T

    # column s = 0 holds only the cell (0, 0)
    prev2 = C.new_full((B, T + 1), big)  # column s-2
    prev1 = C.new_full((B, T + 1), big)  # column s-1
    prev1[:, 0] = 0.0
    if subsequence:
        # Entering at any expert index is free, i.e. R[0, j] = 0 for every j,
        # and row 0 of every skewed column is therefore 0.
        row0 = C.new_zeros(B)
    else:
        row0 = None

    last_row_terms = []  # R[T, j] for j = 1..L, collected for the open-end case
    for sdiag in range(1, T + L + 1):
        j_idx = sdiag - i_idx  # expert column for each agent row on this diagonal
        valid = (j_idx >= 1) & (j_idx <= L)

        # the three predecessors: R[i-1, j] and R[i, j-1] live in column s-1,
        # R[i-1, j-1] in column s-2
        up = prev1[:, 0:T]  # R[i-1, j]   for i = 1..T
        left = prev1[:, 1 : T + 1]  # R[i, j-1]   for i = 1..T
        diag = prev2[:, 0:T]  # R[i-1, j-1] for i = 1..T

        stacked = torch.stack((up, left, diag), dim=-1)
        softmin = -gamma * torch.logsumexp(-stacked / gamma, dim=-1)

        cost = C[:, i_idx - 1, (j_idx - 1).clamp(0, L - 1)]
        cur_body = cost + softmin
        cur_body = torch.where(valid.unsqueeze(0).expand(B, T), cur_body, cur_body.new_full((), big))

        # row 0 of this column: free entry for subsequence, otherwise unreachable
        if row0 is not None:
            head = row0.unsqueeze(-1)
        else:
            head = cur_body.new_full((B, 1), big)
        cur = torch.cat((head, cur_body), dim=-1)

        if sdiag - T >= 1 and sdiag - T <= L:
            last_row_terms.append(cur[:, T])

        prev2, prev1 = prev1, cur

    if open_end:
        # Finishing at any expert index is free: soft minimum over R[T, 1..L].
        stacked_last = torch.stack(last_row_terms, dim=-1)
        value = -gamma * torch.logsumexp(-stacked_last / gamma, dim=-1)
    else:
        value = prev1[:, T]

    # The alignment matrix is exactly dvalue/dC; autograd gives it to us without
    # a hand-written backward recursion, and it is correct by construction.
    (W,) = torch.autograd.grad(value.sum(), C, create_graph=C.requires_grad)
    return value, W


class SoftDTWCriterion(BestOfKCriterion):
    """Ordered-alignment counterpart of :class:`OTSinkhornCriterion`.

    Uses **no Sinkhorn and no optimal transport**: the alignment is the soft-DTW
    dynamic program. It shares only the ground cost with the OT criterion, via
    the module-level :func:`ground_cost_matrix`.

    Everything outside the alignment itself -- ground cost, per-step cost
    construction, best-of-K softmin, reward transformation, critic -- is shared
    with the OT criterion, so swapping the two isolates the alignment objective.

    ``normalize`` controls how the soft alignment matrix is turned into per-step
    weights, which fixes the cost scale:

    - ``"rows"`` (default): each agent step carries mass ``1/T``, exactly the OT
      plan's agent-side marginal. Per-step costs are then on the same scale as
      ``OTSinkhornCriterion``'s, so a swap isolates the alignment objective.
    - ``"total"``: keep soft-DTW's own row masses, rescaled to unit total mass.
    - ``"none"``: raw ``dvalue/dC``; scores are path costs and are NOT
      comparable to the OT criterion's.

    With ``"none"`` or ``"total"`` the best-of-K temperature must be re-matched
    to the resulting score scale rather than copied from the OT configuration,
    or the comparison measures temperature rather than alignment.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        cost_type: Literal["huber", "l2", "cosine"] = "l2",
        huber_delta: float = 1.0,
        open_end: bool = False,
        subsequence: bool = False,
        normalize: Literal["rows", "total", "none"] = "rows",
    ):
        super().__init__()
        self.gamma = gamma
        self.cost_type = cost_type
        self.huber_delta = huber_delta
        self.open_end = open_end
        self.subsequence = subsequence
        if normalize not in ("rows", "total", "none"):
            raise ValueError(f"Invalid normalize: {normalize}")
        self.normalize = normalize

    def forward(self, sim_f: torch.Tensor, exp_f: torch.Tensor):
        B, K, T, d = exp_f.shape
        sim_f_tiled = sim_f.unsqueeze(1).expand(B, K, T, d).contiguous().view(B * K, T, d)
        exp_f_flat = exp_f.contiguous().view(B * K, T, d)

        # Same ground cost as the OT criterion, via the shared free function.
        # No Sinkhorn anywhere on this path: the alignment is the DP recursion
        # in soft_dtw() below.
        C, _ = ground_cost_matrix(
            sim_f_tiled, exp_f_flat, cost_type=self.cost_type, huber_delta=self.huber_delta
        )  # [B*K, T, L]

        # soft_dtw needs grad w.r.t. C to read off the alignment matrix.
        C_for_align = C.detach().requires_grad_(True)
        with torch.enable_grad():
            _, W = soft_dtw(
                C_for_align,
                gamma=self.gamma,
                open_end=self.open_end,
                subsequence=self.subsequence,
            )
        W = W.detach()
        if self.normalize == "rows":
            # Match the OT plan's agent-side marginal exactly: each agent step
            # carries mass 1/T. The per-step costs are then on an identical
            # scale to OTSinkhornCriterion's, so the only remaining difference
            # between the two criteria is *which* expert steps each agent step
            # is matched to -- a monotone path versus an unconstrained coupling.
            # This is the controlled comparison the reviewers asked for.
            W = W / W.sum(dim=-1, keepdim=True).clamp_min(1e-12) / T
        elif self.normalize == "total":
            # Preserve soft-DTW's own per-row mass (an agent step matched to
            # several expert steps keeps more weight), rescaled so the whole
            # alignment carries unit mass like the OT plan.
            W = W / W.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

        # Same shape of computation as OTSinkhornCriterion: fixed alignment,
        # gradients through the costs.
        plan_costs = (W * C).sum(dim=-1).view(B, K, T)
        Lk = plan_costs.sum(dim=-1)  # [B, K]
        info = {
            "per_step_costs": plan_costs,
            "transport_plans": W.view(B, K, T, T),
        }
        return Lk, info


def ordered_coverage(
    C: torch.Tensor,
    lam: float = 1.0,
    normalize_p: bool = False,
    return_log: bool = False,
) -> torch.Tensor:
    """ORCA-style ordered-coverage reward (Zhang et al., ICML 2025).

    Reviewer Q9cS names ORCA as a missing baseline. Its reward is a coverage
    dynamic program rather than an alignment path:

        P_{t,j}  proportional to  exp(-lam * d(o_t, o~_j))
        C_{t,j}  =  max( C_{t-1,j},  C_{t,j-1} * P_{t,j} )
        R_t      =  C_{t,L-2} * P_{t,L-1}

    C_{t,j} is the probability that by time t the agent has occupied
    demonstration frame j *having already covered every earlier frame in
    order*, so reward accrues only once the whole prefix is covered in
    sequence. That is a strictly stronger ordering requirement than soft-DTW,
    which only needs a monotone path, and much stronger than OT, which imposes
    no ordering at all.

    Fully differentiable in ``C``: the recursion is max-plus in log space and
    both ``maximum`` and ``logsumexp`` pass subgradients, so the policy
    receives gradient through the ground cost -- unlike the OT path, no part of
    this is detached.

    Evaluated with the same anti-diagonal wavefront as ``soft_dtw``: cell
    (t, j) depends on (t-1, j) and (t, j-1), both of which lie on the previous
    anti-diagonal, so cells sharing t+j are independent and the T*L sequential
    steps collapse to T+L vectorised ones.

    ⚠️ Deviations from the published method. State these whenever this is
    reported -- a negative result for an adapted method is not a negative
    result for ORCA:
      * the original uses visual embeddings (ResNet50 + cosine on Meta-World, a
        joint-position predictor on Humanoid); here ``d`` is our normalised
        state-transition cost. The Humanoid setup is the closer analogue.
      * the original appends a timestep-percentage feature to the policy input;
        we do not, since that would change the observation space relative to
        every other variant and confound the comparison.
      * the original learns from one demonstration with DrQ-v2/SAC; here it
        scores one candidate segment inside best-of-K on the FOCUS backbone.

    Args:
        C: ``[B, T, L]`` ground cost (agent time x demonstration frame).
        lam: sharpness of the occupancy probability.
        normalize_p: make P a distribution over demonstration frames per agent
            step. Default False. Normalising reads "proportional to" as a
            distribution, which makes the coverage a product of ~L terms of
            size ~1/L; at L = 32 that underflows float32 to exactly zero for
            any agent not already close to the demonstration, killing the
            gradient. The unnormalised kernel exp(-lam*d) is bounded in (0, 1]
            and does not collapse when the match is good.
        return_log: return log-coverage instead of coverage. Strongly preferred
            inside a differentiable pipeline -- see ORCACriterion.

    Returns:
        ``[B, T]`` per-step reward in [0, 1], or its log if ``return_log``.
    """
    B, T, L = C.shape
    logP = -lam * C
    if normalize_p:
        logP = logP - torch.logsumexp(logP, dim=-1, keepdim=True)

    NEG = -1e9
    device = C.device
    t_idx = torch.arange(T, device=device)

    # Skewed coordinates s = t + j. Column s holds, for each agent row t, the
    # cell (t, s - t). prev1 is column s-1.
    prev1 = C.new_full((B, T + 1), NEG)   # row r corresponds to agent step r-1
    log_rewards = [None] * T

    for sdiag in range(0, T + L):
        j_idx = sdiag - t_idx                              # [T]
        valid = (j_idx >= 0) & (j_idx <= L - 1)
        jg = j_idx.clamp(0, L - 1)
        lp = logP[:, t_idx, jg]                            # [B, T]

        up = prev1[:, 0:T]        # C_{t-1, j}   (row t-1 of column s-1)
        left = prev1[:, 1:T + 1]  # C_{t, j-1}   (row t   of column s-1)
        # j == 0 has an empty prefix, so C_{t,-1} = 1 -> log 0 contribution
        left = torch.where((j_idx == 0).unsqueeze(0).expand(B, T), torch.zeros_like(left), left)
        cur_body = torch.maximum(up, left + lp)
        cur_body = torch.where(valid.unsqueeze(0).expand(B, T), cur_body,
                               torch.full_like(cur_body, NEG))

        head = C.new_full((B, 1), NEG)   # row 0 = agent step -1, nothing covered
        cur = torch.cat((head, cur_body), dim=-1)

        # R_t = C_{t, L-2} * P_{t, L-1}: available on the diagonal where
        # j = L-2, i.e. s = t + L - 2.
        t_done = sdiag - (L - 2)
        if L >= 2 and 0 <= t_done < T:
            log_rewards[t_done] = cur[:, t_done + 1] + logP[:, t_done, L - 1]

        prev1 = cur

    for t in range(T):
        if log_rewards[t] is None:
            log_rewards[t] = C.new_full((B,), NEG)
    log_R = torch.stack(log_rewards, dim=1).clamp_min(NEG)
    return log_R if return_log else log_R.exp()


class ORCACriterion(BestOfKCriterion):
    """Ordered-coverage criterion, a drop-in alongside OT and soft-DTW.

    Shares the ground cost with every other criterion via
    :func:`ground_cost_matrix`, so swapping it in changes only how agent steps
    are scored against demonstration frames. Read the deviation list in
    :func:`ordered_coverage` before reporting anything from it.
    """

    def __init__(
        self,
        lam: float = 0.25,
        cost_type: Literal["huber", "l2", "cosine"] = "l2",
        huber_delta: float = 1.0,
        normalize_p: bool = False,
        cost_mode: Literal["neg_log_r", "one_minus_r"] = "neg_log_r",
    ):
        super().__init__()
        self.lam = lam
        self.cost_type = cost_type
        self.huber_delta = huber_delta
        self.normalize_p = normalize_p
        self.cost_mode = cost_mode

    def forward(self, sim_f: torch.Tensor, exp_f: torch.Tensor):
        B, K, T, d = exp_f.shape
        sim_t = sim_f.unsqueeze(1).expand(B, K, T, d).contiguous().view(B * K, T, d)
        exp_t = exp_f.contiguous().view(B * K, T, d)
        C = ground_cost_matrix(sim_t, exp_t, cost_type=self.cost_type,
                               huber_delta=self.huber_delta)[0]
        # ORCA yields a reward in [0, 1]; this pipeline consumes a non-negative
        # cost. Two conversions, both monotone:
        #
        #   one_minus_r : 1 - R, the literal reading. Unusable here as the sole
        #       option: coverage is a product over the window, so for an agent
        #       far from the demonstration R underflows to exactly 0 in float32
        #       (measured at T = L = 32) and the gradient vanishes. ORCA's own
        #       setting consumes R as a scalar RL reward for SAC/DrQ-v2, where
        #       a near-zero reward merely means "no signal yet" and exploration
        #       still drives progress; differentiating through it is harsher.
        #   neg_log_r   : -log R, computed from the log-domain recursion, so it
        #       never underflows and the gradient stays informative over the
        #       whole range. Default here, and a documented deviation.
        if self.cost_mode == "neg_log_r":
            logR = ordered_coverage(C, lam=self.lam, normalize_p=self.normalize_p,
                                    return_log=True)
            per_step = (-logR).view(B, K, T)
        else:
            R = ordered_coverage(C, lam=self.lam, normalize_p=self.normalize_p)
            per_step = (1.0 - R).view(B, K, T)
        Lk = per_step.sum(dim=-1)
        return Lk, {"per_step_costs": per_step}


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
        mlp_features_dim: Optional[int] = None,  # if set, use MLP features
        feature_dim: Optional[int] = None,
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
        self.mlp_features_dim = mlp_features_dim
        self.feature_dim = feature_dim
        self.return_per_step_costs = return_per_step_costs
        self.device = device

        if self.input_type not in ("state", "state_state"):
            raise ValueError(f"Invalid input_type: {self.input_type}")

        # Features
        if self.mlp_features_dim is not None:
            assert feature_dim is not None, "Set feature_dim when use_mlp_features=True"
            in_dim = feature_dim * 2 if self.input_type == "state_state" else feature_dim
            self.feat = MLPFeat(in_dim, self.mlp_features_dim).to(device=device)
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
        if "transport_plans" in crit_info:
            info["transport_plans"] = crit_info["transport_plans"].detach()

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
    ot_crit = OTSinkhornCriterion(eps=0.1, iters=60, cost_type="l2", huber_delta=1.0)
    loss_fn_ot = BestOfK(
        T=T,
        K=8,
        tau=0.5,
        mlp_features_dim=64,
        feature_dim=d_in,
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
        mlp_features_dim=64,
        feature_dim=d_in,
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
        mlp_features_dim=64,
        feature_dim=d_in,
        return_per_step_costs=True,
        criterion=cos_crit,
    ).to(device)
    loss_cos, info_cos = loss_fn_cos(sim_seq, expert_bank, expert_lens=expert_lens, sim_is_window=False)
    print("Cosine Loss:", loss_cos.item(), "Best (first 5):", info_cos["best_idx"][:5])
