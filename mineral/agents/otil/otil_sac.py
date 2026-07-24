import torch

from ..sac.sac import SAC
from .reward import DetachedOTRewardMixin


class OTILSAC(DetachedOTRewardMixin, SAC):
    """SAC trained only from detached Best-of-K OT trajectory rewards."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        self._ot_reward_horizon = self.sac_config.horizon_len
        self._init_detached_ot_reward()

    def _encode_ot_obs(self, obs):
        batch_shape = next(iter(obs.values())).shape[:2]
        flat_obs = {key: value.reshape(-1, *value.shape[2:]) for key, value in obs.items()}
        encoded = self.encoder(flat_obs)
        encoded = encoded["z"] if isinstance(encoded, dict) else encoded
        return encoded.reshape(*batch_shape, -1)

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        if timesteps != self._ot_reward_horizon:
            raise ValueError(
                "OTIL-SAC needs warm_up == horizon_len so every collected transition "
                "receives a reward from an identically sized OT window"
            )

        traj_obs = {
            key: torch.empty((self.num_actors, timesteps) + shape, dtype=torch.float32, device=self.device)
            for key, shape in self.obs_space.items()
        }
        traj_actions = torch.empty((self.num_actors, timesteps, self.action_dim), device=self.device)
        traj_next_obs = {
            key: torch.empty((self.num_actors, timesteps) + shape, dtype=torch.float32, device=self.device)
            for key, shape in self.obs_space.items()
        }
        traj_dones = torch.empty((self.num_actors, timesteps), device=self.device)

        ot_obs_rms = self._snapshot_ot_normalizer()
        obs_window = self._new_ot_window(self.obs)

        for i in range(timesteps):
            if not self.env_autoresets:
                raise NotImplementedError

            if self.normalize_input:
                for key, value in self.obs.items():
                    self.obs_rms[key].update(value)
            if random:
                actions = torch.rand((self.num_actors, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                actions = self.get_actions(obs=self.obs, sample=sample)

            next_obs_raw, env_rewards, dones, infos = env.step(actions)
            next_obs = self._convert_obs(next_obs_raw)
            dones = torch.as_tensor(dones, device=self.device)
            env_rewards = torch.as_tensor(env_rewards, device=self.device)

            real_next_obs_raw = infos.get("obs_before_reset", next_obs_raw)
            if real_next_obs_raw is None:
                real_next_obs_raw = next_obs_raw
            self._append_ot_window(obs_window, self._convert_obs(real_next_obs_raw))

            done_indices = torch.where(dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, env_rewards, done_indices, infos)
            replay_dones = self._handle_timeout(dones, infos) if self.sac_config.handle_timeout else dones

            for key, value in self.obs.items():
                traj_obs[key][:, i] = value
            traj_actions[:, i] = actions
            traj_dones[:, i] = replay_dones
            for key, value in next_obs.items():
                traj_next_obs[key][:, i] = value
            self.obs = next_obs

        self.metrics.flush_video(self.epoch)

        traj_rewards = self._compute_detached_ot_rewards(obs_window, ot_obs_rms).unsqueeze(-1)
        traj_dones = traj_dones.unsqueeze(-1)
        data = self.n_step_buffer.add_to_buffer(
            traj_obs,
            traj_actions,
            traj_rewards,
            traj_next_obs,
            traj_dones,
        )
        return data, timesteps * self.num_actors

    def update_net(self, memory):
        results = super().update_net(memory)
        for key, value in self._last_ot_reward_stats.items():
            results[f"ot/{key}"].append(value)
        return results
