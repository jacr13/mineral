import torch

from ...common.demos import get_demos
from ..sac.sac import SAC
from .rewarder import PWILRewarder


class PWIL(SAC):
    """Primal Wasserstein Imitation Learning, built on SAC.

    Reference: Dadashi, Hussenot, Geist, Pietquin, "Primal Wasserstein Imitation
    Learning" (https://arxiv.org/abs/2006.04678), code at
    https://github.com/google-research/google-research/tree/master/pwil

    The reward is a fixed, non-adversarial function of the (greedy, primal)
    Wasserstein distance between the agent's rollout and a pool of expert
    (state, action) atoms (`PWILRewarder`); there is no learned discriminator
    or critic-facing on-policy requirement of any kind. The original paper
    pairs this reward with D4PG, an off-policy actor-critic, precisely because
    nothing about the reward needs fresh on-policy data to stay valid, unlike
    GAIL's discriminator. This class follows that design on top of this
    repo's SAC (matching `DAC`/`OPOLO`'s own choice of off-policy backbone):
    the PWIL reward is computed once per environment step and written
    straight into the replay buffer, so `explore_env` is the only thing
    overridden here; everything else (critic/actor updates, replay buffer,
    target networks) is plain `SAC`, unchanged.
    """

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)

        self.pwil_config = full_cfg.agent.get("pwil", {})

        demos_config = self.pwil_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)

        time_horizon = self.pwil_config.get("time_horizon", None)
        if time_horizon is None:
            time_horizon = self.env.max_episode_length
        self.observation_only = bool(self.pwil_config.get("observation_only", False))
        # With state-only atoms the reward r(s_t) computed before stepping does not depend on the executed
        # action a_t, so credit for a_t only reaches the critic through bootstrapping. Score the state the
        # action leads to (s_{t+1}, before any auto-reset) instead. Defaults to on for state-only atoms; with
        # (s, a) atoms the reward already depends on a_t and is computed on the pre-step pair.
        reward_on_next_obs = self.pwil_config.get("reward_on_next_obs", None)
        self.reward_on_next_obs = self.observation_only if reward_on_next_obs is None else bool(reward_on_next_obs)
        if self.reward_on_next_obs and not self.observation_only:
            raise ValueError("pwil.reward_on_next_obs requires pwil.observation_only=true")

        self.pwil_rewarder = PWILRewarder(
            self.demos,
            num_envs=self.num_actors,
            device=self.device,
            time_horizon=time_horizon,
            alpha=float(self.pwil_config.get("alpha", 5.0)),
            beta=float(self.pwil_config.get("beta", 5.0)),
            observation_only=self.observation_only,
        )
        self._last_pwil_stats = {}

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        # Identical to SAC.explore_env, except rewards come from the PWIL
        # rewarder instead of the environment, and each environment's
        # expert-atom budget is reset when its episode ends (`PWILRewarder`
        # persists across calls, so budgets correctly carry over between
        # `explore_env` calls within the same episode).
        traj_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_actions = torch.empty((self.num_actors, timesteps) + (self.action_dim,), device=self.device)
        traj_rewards = torch.empty((self.num_actors, timesteps), device=self.device)
        traj_next_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.obs_space.items()
        }
        traj_dones = torch.empty((self.num_actors, timesteps), device=self.device)

        for i in range(timesteps):
            if not self.env_autoresets:
                raise NotImplementedError

            if self.normalize_input:
                for k, v in self.obs.items():
                    self.obs_rms[k].update(v)
            if random:
                actions = torch.rand((self.num_actors, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                actions = self.get_actions(obs=self.obs, sample=sample)

            if not self.reward_on_next_obs:
                # PWIL reward from the pre-step observation and the executed action.
                shaped_rewards = self.pwil_rewarder.compute_reward(self.obs, actions)

            next_obs_raw, rewards, dones, infos = env.step(actions)
            next_obs = self._convert_obs(next_obs_raw)

            if self.reward_on_next_obs:
                # Reward the state reached by `actions`. `obs_before_reset` is the true s_{t+1} for envs that
                # just auto-reset (`next_obs` is then the first observation of the new episode).
                real_next_obs_raw = infos.get("obs_before_reset", None)
                if real_next_obs_raw is None:
                    real_next_obs_raw = next_obs_raw
                shaped_rewards = self.pwil_rewarder.compute_reward(self._convert_obs(real_next_obs_raw), actions)

            done_indices = torch.where(dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards, done_indices, infos)

            if len(done_indices) > 0:
                self.pwil_rewarder.reset(env_ids=done_indices)

            if self.sac_config.handle_timeout:
                dones = self._handle_timeout(dones, infos)

            for k, v in self.obs.items():
                traj_obs[k][:, i] = v
            traj_actions[:, i] = actions
            traj_dones[:, i] = dones
            traj_rewards[:, i] = shaped_rewards
            for k, v in next_obs.items():
                traj_next_obs[k][:, i] = v
            self.obs = next_obs  # update obs

        self.metrics.flush_video(self.epoch)

        self._last_pwil_stats = {
            "reward_mean": traj_rewards.mean(),
            "reward_std": traj_rewards.std(unbiased=False),
            "expert_return": torch.tensor(float(self.demos["expert_return"]), device=self.device),
        }

        traj_rewards = self.reward_shaper(traj_rewards.reshape(self.num_actors, timesteps, 1))
        traj_dones = traj_dones.reshape(self.num_actors, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones)

        return data, timesteps * self.num_actors

    def update_net(self, memory):
        results = super().update_net(memory)
        for key, value in self._last_pwil_stats.items():
            results[f"pwil/{key}"].append(value)
        return results
