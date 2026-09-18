import torch


class PWILRewarder:
    """Vectorized, batched port of the PWIL rewarder.

    Reference implementation (single-env, numpy/sklearn):
    https://github.com/google-research/google-research/tree/master/pwil
    (``pwil/rewarder.py``, from Dadashi et al., "Primal Wasserstein Imitation
    Learning", https://arxiv.org/abs/2006.04678).

    PWIL builds a pool of expert (observation, action) "atoms", each carrying an
    equal probability mass of ``1 / num_atoms``. For every environment step, it
    greedily consumes ``1 / time_horizon`` worth of mass from the nearest
    remaining expert atom(s) -- this is the greedy/primal approximation to the
    optimal-transport coupling between the agent's occupancy measure and the
    expert's. The accumulated transport cost is turned into a reward via
    ``alpha * exp(-beta * T / sqrt(dim) * cost)``. There is nothing learned here:
    the reward is a fixed function of the (standardized) distance to expert
    demonstrations, recomputed and depleted online as each episode unrolls.

    This class supports many parallel environments at once: each environment
    keeps its own remaining expert-mass budget (``remaining_weights``), reset
    independently whenever that environment's episode ends, while all
    environments share the same underlying (fixed) pool of expert atoms.
    """

    def __init__(
        self,
        demos,
        num_envs,
        device,
        time_horizon,
        alpha=5.0,
        beta=5.0,
        observation_only=False,
    ):
        """Builds the expert atom pool and per-env bookkeeping.

        Args:
            demos: dict as returned by ``mineral.common.demos.get_demos``, with
                "obs" (dict of tensors [n_trajs, T, ...]) and "act"
                ([n_trajs, T, act_dim]). Random per-trajectory subsampling
                offsets (as in the original ``filter_demonstrations``) are
                already applied upstream by ``get_demos``.
            num_envs: number of parallel environments (rows of per-env state).
            device: torch device to keep all rewarder state on.
            time_horizon: task horizon T used both for the per-step expert-mass
                budget (1/T) and for the reward kernel width.
            alpha: reward scale.
            beta: reward kernel width factor.
            observation_only: if True, atoms/reward ignore actions entirely.
        """
        self.device = device
        self.observation_only = observation_only
        self.time_horizon = float(time_horizon)
        self.alpha = float(alpha)
        self.num_envs = num_envs

        obs = demos["obs"]
        act = demos["act"]
        n_trajs, traj_len = act.shape[0], act.shape[1]

        # Pool all (subsampled) expert transitions from every selected trajectory
        # into a single flat set of atoms, exactly as the original
        # `filter_demonstrations`/`vectorize` do.
        flat_obs = self._flatten(obs, num_leading=2).reshape(n_trajs * traj_len, -1)
        dim_obs = flat_obs.shape[-1]
        if observation_only:
            atoms = flat_obs
            dim_act = 0
        else:
            flat_act = act.reshape(n_trajs * traj_len, -1)
            dim_act = flat_act.shape[-1]
            atoms = torch.cat([flat_obs, flat_act], dim=-1)
        atoms = atoms.to(device=device, dtype=torch.float32)

        # Standardized Euclidean distance (mirrors sklearn's StandardScaler,
        # fit once on the expert atom pool and reused for agent atoms).
        self.atom_mean = atoms.mean(dim=0, keepdim=True)
        self.atom_std = atoms.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        self.expert_atoms = (atoms - self.atom_mean) / self.atom_std

        self.num_atoms = self.expert_atoms.shape[0]
        self.expert_weight_init = 1.0 / self.num_atoms
        self.reward_sigma = beta * self.time_horizon / (max(dim_obs + dim_act, 1) ** 0.5)

        # Per-env remaining expert mass, one row per parallel environment.
        self.remaining_weights = torch.empty(num_envs, self.num_atoms, device=device)
        self.reset()

    @staticmethod
    def _flatten(obs_dict, num_leading):
        """Concatenates a dict of tensors into one flat feature vector.

        Keeps the first ``num_leading`` dims (e.g. batch, or batch+time) intact
        and flattens everything else per key, in sorted key order so demo and
        live observations are always laid out identically.
        """
        if not isinstance(obs_dict, dict):
            value = obs_dict
            return value.reshape(*value.shape[:num_leading], -1)
        parts = []
        for key in sorted(obs_dict.keys()):
            value = obs_dict[key]
            parts.append(value.reshape(*value.shape[:num_leading], -1))
        return torch.cat(parts, dim=-1)

    def reset(self, env_ids=None):
        """Refills the expert-mass budget for the given envs (or all envs)."""
        if env_ids is None:
            self.remaining_weights.fill_(self.expert_weight_init)
        else:
            if torch.is_tensor(env_ids):
                env_ids = env_ids.to(device=self.remaining_weights.device, dtype=torch.long)
            elif len(env_ids) == 0:
                return
            self.remaining_weights[env_ids] = self.expert_weight_init

    @torch.no_grad()
    def compute_reward(self, obs, actions):
        """Computes the PWIL reward for one transition per environment.

        This implements Algorithm 1 of the paper, vectorized over environments:
        for each env, greedily assign its per-step expert-mass budget (1/T) to
        the nearest remaining expert atom(s), accumulate the transport cost, and
        map it through the exponential reward kernel. Each env's remaining
        expert mass is depleted in place, so successive calls within an episode
        (with no intervening `reset`) correctly consume the same env's budget.

        Args:
            obs: dict of observation tensors, each shaped [num_envs, ...].
            actions: tensor shaped [num_envs, act_dim] (ignored if
                ``observation_only``).

        Returns:
            reward: tensor shaped [num_envs].
        """
        flat_obs = self._flatten(obs, num_leading=1).to(device=self.device, dtype=torch.float32)
        if self.observation_only:
            atom = flat_obs
        else:
            flat_act = actions.reshape(actions.shape[0], -1).to(device=self.device, dtype=torch.float32)
            atom = torch.cat([flat_obs, flat_act], dim=-1)
        atom = (atom - self.atom_mean) / self.atom_std

        # [num_envs, num_atoms] standardized Euclidean distances.
        dists = torch.cdist(atom.unsqueeze(0), self.expert_atoms.unsqueeze(0)).squeeze(0)

        # Per-step expert-mass budget; the tiny epsilon avoids the greedy loop
        # spinning on floating-point residue once mass is (almost) exhausted,
        # exactly as in the reference implementation.
        weight_budget = torch.full((self.num_envs,), 1.0 / self.time_horizon - 1e-6, device=self.device)
        cost = torch.zeros(self.num_envs, device=self.device)

        # Greedily consume the nearest remaining expert atom(s) until every
        # env's budget is exhausted. Bounded by num_atoms + 1 iterations (worst
        # case: one atom fully consumed per iteration), but in practice this
        # breaks out after very few iterations since 1/T << 1/num_atoms.
        for _ in range(self.num_atoms + 1):
            active = weight_budget > 1e-12
            if not torch.any(active):
                break

            masked_dists = torch.where(self.remaining_weights > 0, dists, dists.new_full((), float("inf")))
            argmin_idx = masked_dists.argmin(dim=1, keepdim=True)  # [num_envs, 1]
            atom_dist = dists.gather(1, argmin_idx).squeeze(1)
            atom_weight = self.remaining_weights.gather(1, argmin_idx).squeeze(1)

            take = torch.where(active, torch.minimum(weight_budget, atom_weight), torch.zeros_like(weight_budget))

            cost = cost + take * atom_dist
            weight_budget = weight_budget - take

            new_weight = atom_weight - take
            self.remaining_weights.scatter_(1, argmin_idx, new_weight.unsqueeze(1))

        reward = self.alpha * torch.exp(-self.reward_sigma * cost)
        return reward
