import torch

from ..ppo.ppo import PPO
from .reward import DetachedOTRewardMixin


class OTILPPO(DetachedOTRewardMixin, PPO):
    """PPO trained only from detached Best-of-K OT trajectory rewards."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        self._ot_reward_horizon = self.horizon_len
        self._init_detached_ot_reward()

    def _encode_ot_obs(self, obs):
        model = self.model.module if hasattr(self.model, "module") else self.model
        batch_shape = next(iter(obs.values())).shape[:2]
        flat_obs = {key: value.reshape(-1, *value.shape[2:]) for key, value in obs.items()}
        return model._encode(flat_obs).reshape(*batch_shape, -1)

    @torch.no_grad()
    def play_steps(self):
        ot_obs_rms = self._snapshot_ot_normalizer()
        obs_window = self._new_ot_window(self.obs)
        timeout_bootstrap = torch.zeros(
            (self.horizon_len, self.num_actors, 1),
            dtype=torch.float32,
            device=self.device,
        )

        for n in range(self.horizon_len):
            if not self.env_autoresets:
                if any(self.dones):
                    done_indices = torch.where(self.dones)[0].tolist()
                    obs_reset = self._convert_obs(self.env.reset_idx(done_indices))
                    for key, value in obs_reset.items():
                        self.obs[key][done_indices] = value

            model_out = self.model_act(self.obs)
            self.storage.update_data("obses", n, self.obs)
            for key in ("actions", "neglogp", "values", "mu", "sigma"):
                self.storage.update_data(key, n, model_out[key])

            actions = torch.clamp(model_out["actions"], -1.0, 1.0)
            next_obs_raw, env_rewards, dones, infos = self.env.step(actions)
            real_next_obs_raw = infos.get("obs_before_reset", next_obs_raw)
            if real_next_obs_raw is None:
                real_next_obs_raw = next_obs_raw
            real_next_obs = self._convert_obs(real_next_obs_raw)
            self._append_ot_window(obs_window, real_next_obs)

            self.obs = self._convert_obs(next_obs_raw)
            env_rewards = torch.as_tensor(env_rewards, device=self.device).reshape(-1, 1)
            self.dones = torch.as_tensor(dones, device=self.device)
            self.storage.update_data("dones", n, self.dones)

            if self.value_bootstrap and "time_outs" in infos:
                time_outs = torch.as_tensor(infos["time_outs"], device=self.device).reshape(-1, 1)
                timeout_bootstrap[n] = self.gamma * model_out["values"] * time_outs.float()

            done_indices = torch.where(self.dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, env_rewards.squeeze(-1), done_indices, infos)
        self.metrics.flush_video(self.epoch)

        ot_rewards = self._compute_detached_ot_rewards(obs_window, ot_obs_rms).transpose(0, 1).unsqueeze(-1)
        for n in range(self.horizon_len):
            self.storage.update_data("rewards", n, ot_rewards[n] + timeout_bootstrap[n])

        last_values = self.model_act(self.obs)["values"]
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        values = self.storage.data_dict["values"]
        returns = self.storage.data_dict["returns"]
        if self.normalize_value:
            self.value_rms.update(values)
            values = self.value_rms.normalize(values)
            self.value_rms.update(returns)
            returns = self.value_rms.normalize(returns)
        self.storage.data_dict["values"] = values
        self.storage.data_dict["returns"] = returns

    def train_epoch(self):
        results = super().train_epoch()
        for key, value in self._last_ot_reward_stats.items():
            results[f"ot/{key}"].append(value)
        return results
