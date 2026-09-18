import torch

from ..otil.best_of_k import log_sinkhorn
from ..pwil.rewarder import PWILRewarder


class OOPSRewarder:
    """Full-episode Sinkhorn optimal-transport reward, as used by OOPS.

    Reference: "Imitation Learning from Observation through Optimal
    Transport" (OOPS), supplementary code in ``OOPS_Supplementary/OOPS``
    (``main.py``'s ``compute_ot`` + the main training loop's per-episode OT
    matching, ``TD3.py``'s ``reward_sigma``).

    Unlike PWIL (which greedily consumes expert mass step by step within an
    episode) or OTIL (which matches a *sliding window* against a best-of-K
    crop of demo trajectories), OOPS computes one *entropic* (Sinkhorn)
    optimal-transport plan between an agent's just-COMPLETED full episode and
    one full expert demonstration trajectory (of the same length), redrawn at
    random for every episode. The reward for each agent timestep is the
    negative transport cost that step accumulates under that plan, so reward
    is only available once a whole episode has been observed -- there is no
    incremental, step-by-step reward the way PWIL/OTIL have.

    Sinkhorn's entropic OT solver is reused unchanged from this repo's own
    OTIL implementation (`mineral.agents.otil.best_of_k.log_sinkhorn`), which
    already assumes uniform marginals over equal-length sequences -- exactly
    OOPS's setting, since both the agent episode and the expert trajectory it
    is matched against always have the same fixed length. This avoids adding
    a dependency on the reference code's own OT solver (Python OT / POT).

    Two independent normalizers are kept, mirroring the reference's own
    design: this class's ``state_mean``/``state_std`` (and, in
    ``state_action`` mode, ``action_mean``/``action_std``) are fit *once* from
    the expert demonstrations and used only to standardize OT atoms (both the
    expert's and the agent's) before computing transport costs. The policy's
    own input normalization (analogous to the reference's bootstrapped
    "combined expert + random-rollout" normalizer) is handled separately by
    the agent via a plain running mean/std, since that is a different concern
    from the OT distance metric.
    """

    def __init__(
        self,
        demos,
        device,
        state_action=False,
        reward_scale=5.0,
        sinkhorn_eps=0.05,
        sinkhorn_iters=1000,
        std_clip=0.33,
        clip_state=True,
        time_horizon=None,
    ):
        """Builds per-trajectory (normalized) expert atom sequences.

        Args:
            demos: dict as returned by ``mineral.common.demos.get_demos``,
                with "obs"/"next_obs" (dicts of tensors [n_trajs, T, ...]) and
                "act" ([n_trajs, T, act_dim]).
            device: torch device to keep all rewarder state on.
            state_action: if True, atoms are (state, action) pairs; if False
                (default, matching the reference's reported runs), atoms are
                (state, next_state) pairs -- i.e. observation-only.
            reward_scale: overall reward scale (`--reward_scale` in the
                reference, default 5.0).
            sinkhorn_eps: entropic regularization strength (`--lamda1`).
            sinkhorn_iters: number of Sinkhorn iterations (`--max_iter`).
            std_clip: floor applied to the expert atom std before dividing
                (`--std_clip`), so near-constant dimensions don't blow up the
                normalized distance.
            clip_state: whether to clamp normalized atoms to [-10, 10]
                (`--clip_state`), matching the reference's optional clip.
            time_horizon: fixed episode length T; defaults to the demos'
                (already-subsampled/trimmed) trajectory length.
        """
        self.device = device
        self.state_action = state_action
        self.reward_scale = float(reward_scale)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.std_clip = float(std_clip)
        self.clip_state = bool(clip_state)

        obs = demos["obs"]
        act = demos["act"].to(device=device, dtype=torch.float32)
        n_trajs, traj_len = act.shape[0], act.shape[1]
        self.num_demos = n_trajs
        self.horizon = int(time_horizon) if time_horizon is not None else traj_len

        flat_obs = PWILRewarder._flatten(obs, num_leading=2).to(device=device, dtype=torch.float32)
        state_dim = flat_obs.shape[-1]

        # Normalizer stats fit once from the expert observations only (mirrors
        # the reference's ExpertDataset, whose mean/std come from the "current
        # obs" side of each demo pair), reused for both obs and next_obs.
        flat_obs_pool = flat_obs.reshape(-1, state_dim)
        self.state_mean = flat_obs_pool.mean(dim=0, keepdim=True)
        self.state_std = flat_obs_pool.std(dim=0, unbiased=True, keepdim=True).clamp_min(self.std_clip)

        if state_action:
            flat_act_pool = act.reshape(-1, act.shape[-1])
            self.action_mean = flat_act_pool.mean(dim=0, keepdim=True)
            self.action_std = flat_act_pool.std(dim=0, unbiased=True, keepdim=True).clamp_min(self.std_clip)
            second = self._normalize(act, self.action_mean, self.action_std)
        else:
            flat_next_obs = PWILRewarder._flatten(demos["next_obs"], num_leading=2).to(device=device, dtype=torch.float32)
            second = self._normalize(flat_next_obs, self.state_mean, self.state_std)

        first = self._normalize(flat_obs, self.state_mean, self.state_std)
        self.demo_atoms = torch.cat([first, second], dim=-1)  # [n_trajs, T, D]

        # Matches the reference's `reward_sigma = reward_scale * T / sqrt(state_dim)`,
        # which always uses the raw state dimension, even in state_action mode.
        self.reward_sigma = self.reward_scale * self.horizon / (state_dim**0.5)

    def _normalize(self, x, mean, std):
        normed = (x - mean) / (3.0 * std + 1e-8)
        if self.clip_state:
            normed = normed.clamp(-10.0, 10.0)
        return normed

    def sample_demo_indices(self, batch_size):
        """Draws a fresh random expert trajectory index per env for the next episode."""
        return torch.randint(0, self.num_demos, (batch_size,), device=self.device)

    @torch.no_grad()
    def compute_episode_reward(self, obs_seq, second_seq, demo_indices):
        """Computes the per-timestep OT reward for a batch of full episodes.

        Args:
            obs_seq: [B, T, D_obs] raw agent observations for one full episode.
            second_seq: [B, T, D_obs] raw next-observations, or [B, T, D_act]
                raw actions if ``state_action``, for the same episodes.
            demo_indices: [B] which expert trajectory each env is matched
                against this episode (see `sample_demo_indices`).

        Returns:
            reward: [B, T] per-agent-timestep reward.
            matching: [B, T] the expert timestep each agent timestep is most
                strongly coupled to under the transport plan (its argmax).
        """
        obs_n = self._normalize(obs_seq.to(torch.float32), self.state_mean, self.state_std)
        if self.state_action:
            second_n = self._normalize(second_seq.to(torch.float32), self.action_mean, self.action_std)
        else:
            second_n = self._normalize(second_seq.to(torch.float32), self.state_mean, self.state_std)
        agent_atoms = torch.cat([obs_n, second_n], dim=-1)  # [B, T, D]

        demo_atoms = self.demo_atoms[demo_indices]  # [B, T, D]

        # sqrt of the (non-negative) Euclidean distance, exactly as in the
        # reference's `M = np.sqrt(cdist(traj1, traj2, metric='euclidean'))`.
        cost = torch.cdist(demo_atoms, agent_atoms, p=2).clamp_min(0.0).sqrt()  # [B, T_demo, T_agent]
        plan = log_sinkhorn(cost, eps=self.sinkhorn_eps, iters=self.sinkhorn_iters)  # [B, T_demo, T_agent]

        weighted_cost = cost * plan
        reward = -self.reward_sigma * weighted_cost.sum(dim=1)  # [B, T_agent]
        matching = plan.argmax(dim=1)  # [B, T_agent], expert index per agent step
        return reward, matching
