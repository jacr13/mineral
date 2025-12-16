import collections
from copy import deepcopy
import torch

from ...common.demos import get_demos
from ..diffrl.shac import SHAC
from .ild import ILD


class DiffMimic(ILD):
    """DiffMimic (Ren et al., 2023) style state-matching with demonstration replay."""

    def __init__(self, full_cfg, **kwargs):
        # Skip ILD.__init__ so we can configure our own loss/demos, but still
        # reuse ILD's actor-only train/update loop.
        SHAC.__init__(self, full_cfg, **kwargs)
        self.diffmimic_config = full_cfg.agent.get("diffmimic", {})

        demos_config = self.diffmimic_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)
        self.expert_lengths = self._compute_expert_lengths()

        self.demo_replay_threshold = self.diffmimic_config.get("demo_replay_threshold", None)
        self.demo_replay_prob = float(self.diffmimic_config.get("demo_replay_prob", 1.0))
        self.normalize_demo = self.diffmimic_config.get("normalize_demo", True)
        self.loss_weights = self.diffmimic_config.get("loss_weights", {})

    # ---- Helpers ----
    def _compute_expert_lengths(self):
        """Return per-trajectory usable lengths from the loaded demonstrations."""
        done = self.demos.get("done", None)
        if done is None:
            obs_any = next(iter(self.demos["obs"].values())) if isinstance(self.demos["obs"], dict) else self.demos["obs"]
            traj_len = obs_any.shape[1]
            return torch.full((obs_any.shape[0],), traj_len, device=self.device, dtype=torch.long)

        lengths = torch.zeros(done.shape[0], device=self.device, dtype=torch.long)
        for i in range(done.shape[0]):
            idx = (done[i] == 1).nonzero(as_tuple=True)[0]
            lengths[i] = idx[0].item() + 1 if len(idx) > 0 else done.shape[1]
        return lengths

    def _gather_expert_window(self, tensor, traj_ids, starts):
        """Gather a [B, T, ...] slice from expert trajectories."""
        B = traj_ids.shape[0]
        time_idx = starts.unsqueeze(-1) + torch.arange(self.horizon_len, device=self.device).view(1, -1)
        idx = traj_ids.view(B, 1).expand(B, self.horizon_len)
        return tensor[idx, time_idx, ...]

    def _sample_expert_batch(self, obs_rms=None):
        """Sample a batch of expert windows aligned with the current horizon."""
        valid_ids = torch.where(self.expert_lengths >= self.horizon_len)[0]
        if len(valid_ids) == 0:
            raise ValueError(f"No expert trajectory long enough for horizon={self.horizon_len}")

        traj_ids = valid_ids[torch.randint(0, len(valid_ids), (self.num_envs,), device=self.device)]
        max_starts = (self.expert_lengths[traj_ids] - self.horizon_len).clamp_min(0)
        starts = torch.floor(torch.rand_like(max_starts, dtype=torch.float32) * (max_starts + 1).float()).long()

        def _process(obs_dict):
            if isinstance(obs_dict, dict):
                gathered = {k: self._gather_expert_window(v, traj_ids, starts) for k, v in obs_dict.items()}
                if obs_rms is not None and self.normalize_demo:
                    gathered = {k: obs_rms[k].normalize(v) for k, v in gathered.items()}
                return gathered
            gathered = self._gather_expert_window(obs_dict, traj_ids, starts)
            if obs_rms is not None and self.normalize_demo:
                gathered = obs_rms.normalize(gathered)
            return gathered

        expert_windows = _process(self.demos["obs"])
        return expert_windows

    def _weighted_mse(self, sim_obs, exp_obs):
        """Compute per-env weighted MSE and per-step costs."""
        def _prepare(t):
            return t if t.dim() >= 3 else t.unsqueeze(1)

        if isinstance(sim_obs, dict):
            per_step_terms = []
            per_env_terms = None
            for k, sim in sim_obs.items():
                sim = _prepare(sim)
                exp = _prepare(exp_obs[k])
                diff = sim - exp
                per_step = diff.view(diff.shape[0], diff.shape[1], -1).pow(2).mean(dim=-1)
                weight = float(self.loss_weights.get(k, 1.0))
                per_step_terms.append(weight * per_step)
                per_env = per_step.mean(dim=1) * weight
                per_env_terms = per_env if per_env_terms is None else per_env_terms + per_env
            per_step_costs = torch.stack(per_step_terms, dim=0).sum(dim=0)
            return per_env_terms, per_step_costs

        sim_obs = _prepare(sim_obs)
        exp_obs = _prepare(exp_obs)
        diff = sim_obs - exp_obs
        per_step_costs = diff.view(diff.shape[0], diff.shape[1], -1).pow(2).mean(dim=-1)
        per_env_cost = per_step_costs.mean(dim=1)
        return per_env_cost, per_step_costs

    def _maybe_demo_replay(self, normalized_obs, expert_step):
        """Optionally replace states for the imitation loss when too far from the demo."""
        if self.demo_replay_threshold is None:
            return normalized_obs, torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        _, per_step_costs = self._weighted_mse(normalized_obs, expert_step)
        per_env_err = per_step_costs.mean(dim=1)
        trigger = per_env_err > self.demo_replay_threshold
        if self.demo_replay_prob < 1.0:
            rand_mask = torch.rand_like(per_env_err) < self.demo_replay_prob
            trigger = trigger & rand_mask

        if isinstance(normalized_obs, dict):
            replaced = {}
            for k, v in normalized_obs.items():
                exp = expert_step[k]
                view_shape = (v.shape[0],) + (1,) * (v.dim() - 1)
                mask = trigger.view(view_shape)
                replaced[k] = torch.where(mask, exp, v)
        else:
            view_shape = (normalized_obs.shape[0],) + (1,) * (normalized_obs.dim() - 1)
            mask = trigger.view(view_shape)
            replaced = torch.where(mask, expert_step, normalized_obs)

        return replaced, trigger.float()

    # ---- Core (replaces ILD.compute_actor_loss) ----
    def compute_actor_loss(self):
        with torch.no_grad():
            obs_rms = deepcopy(self.obs_rms) if self.obs_rms is not None else None

        expert_windows = self._sample_expert_batch(obs_rms=obs_rms)
        if not isinstance(expert_windows, dict):
            expert_windows = {"obs": expert_windows}

        obs = self.env.initialize_trajectory()
        obs = self._convert_obs(obs)

        obs_window = {k: [] for k in obs.keys()}
        replay_events = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        if obs_rms is not None:
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_rms[k].update(v)
            obs = {k: obs_rms[k].normalize(v) for k, v in obs.items()}

        for i in range(self.horizon_len):
            with torch.no_grad():
                for k, v in obs.items():
                    self.obs_buf[k][i] = v.clone()

            z = self.actor_encoder(obs)
            actions, mu, sigma, distr = self.get_actions(obs, z=z, sample=True, dist=True)

            with torch.no_grad():
                self.action_buf[i] = actions.clone()
                self.mus[i, ...] = mu.clone()
                self.sigmas[i, ...] = sigma.clone()

            obs_next, rew, done, info = self.env.step(actions)
            real_obs = info.get("obs_before_reset", obs_next)
            real_obs = obs_next if real_obs is None else real_obs

            obs_next = self._convert_obs(obs_next)
            real_obs = self._convert_obs(real_obs)

            with torch.no_grad():
                self.episode_rewards += rew
                self.episode_lengths += 1

            if obs_rms is not None:
                with torch.no_grad():
                    for k, v in obs_next.items():
                        self.obs_rms[k].update(v)
                obs_next = {k: obs_rms[k].normalize(v) for k, v in obs_next.items()}
            real_obs = {k: obs_rms[k].normalize(v) for k, v in real_obs.items()}

            # demonstration replay anchors the sequence used for the loss
            replayed_obs, replay_flags = self._maybe_demo_replay(
                real_obs,
                {k: v[:, i] for k, v in expert_windows.items()},
            )
            replay_events += replay_flags

            for k, v in replayed_obs.items():
                obs_window[k].append(v)

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_env_ids) > 0:
                done_env_ids = done_env_ids.detach().cpu()
                terminal_obs = info.get("obs_before_reset", None)
                terminal_obs = obs_next if terminal_obs is None else terminal_obs
                terminal_obs = self._convert_obs(terminal_obs)
                if obs_rms is not None:
                    terminal_obs = {k: obs_rms[k].normalize(v) for k, v in terminal_obs.items()}
                self.episode_rewards_tracker.update(self.episode_rewards[done_env_ids])
                self.episode_lengths_tracker.update(self.episode_lengths[done_env_ids])
                self.num_episodes += len(done_env_ids)
                for done_env_id in done_env_ids:
                    self.episode_rewards_hist.append(self.episode_rewards[done_env_id].item())
                    self.episode_lengths_hist.append(self.episode_lengths[done_env_id].item())
                    self.episode_rewards[done_env_id] = 0.0
                    self.episode_lengths[done_env_id] = 0

            if i < self.horizon_len - 1:
                self.done_mask[i] = done.to(dtype=torch.float32)
            else:
                self.done_mask[i] = torch.ones_like(done, dtype=torch.float32)

            obs = obs_next

        obs_window = {k: torch.stack(v, dim=1) for k, v in obs_window.items()}
        per_env_cost, per_step_costs = self._weighted_mse(obs_window, expert_windows)
        actor_loss = per_env_cost.mean()
        info = {
            "imitation_loss": actor_loss.detach(),
            "per_step_costs": per_step_costs.detach(),
            "per_step_cost_mean": per_step_costs.mean().detach(),
            "replay_fraction": (replay_events / max(1, self.horizon_len)).mean().detach(),
            "expert_return": self.demos.get("expert_return", 0.0),
        }

        self.agent_steps += self.horizon_len * self.num_envs
        return actor_loss, info
