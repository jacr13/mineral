import collections
import os
import re
from copy import deepcopy

import torch
import torch.nn as nn

from ...buffers import NStepReplay, ReplayBuffer
from ...common import normalizers
from ...common.demos import get_demos
from ...common.reward_shaper import RewardShaper
from ..agent import Agent
from ..ddpg import models
from ..ddpg.noise import add_normal_noise
from ..ddpg.utils import soft_update
from .rewarder import OOPSRewarder


class OOPS(Agent):
    """Imitation Learning from Observation through Optimal Transport (OOPS).

    Reference: supplementary code in ``OOPS_Supplementary/OOPS`` (builds on
    TD3, https://github.com/sfujim/TD3, and PAL/LAP,
    https://github.com/sfujim/LAP-PAL).

    OOPS trains an off-policy, deterministic actor-critic (TD3) purely from a
    reward derived by matching each just-completed episode against one
    randomly-drawn expert demonstration trajectory via entropic optimal
    transport (Sinkhorn), see `OOPSRewarder`. Three things distinguish it
    from this repo's other reward-shaping baselines (GAIL/DAC/PWIL/OPOLO),
    all of which reuse an *existing* backbone unchanged:

      1. The reward is only defined per *completed episode*, not per step, so
         `explore_env` must collect one full episode before any reward can be
         assigned (no incremental reward the way PWIL's greedy matching or
         GAIL's discriminator give one).
      2. The actor and critic are conditioned on the fraction of the episode
         remaining ("time-to-horizon"), an explicit non-stationarity input
         the reference's own `Actor`/`Critic` take as a second forward
         argument alongside the state -- not something any existing network
         class in this repo's `nets`/`sac`/`ddpg` modules exposes.
      3. The critic loss is PAL (from LAP/PAL, Fujimoto et al.), not MSE: a
         Huber-like loss whose overall scale is normalized by the current
         batch's max TD-error, rather than an unweighted mean-squared error.

    Because of (2), this class cannot simply subclass `DDPG` and override
    `explore_env`/`update_actor`/`update_critic` the way `PWIL`/`DAC` do on
    top of `SAC`: `DDPG.__init__` builds its `Actor`/`EnsembleQ` with
    `state_dim = obs_dim`, one dimension short of what OOPS needs
    (`obs_dim + 1`, or `+ 2` if the optional "match" critic feature is
    enabled). This class therefore subclasses the abstract `Agent` directly
    and reconstructs that piece of `DDPG.__init__` itself, but otherwise
    reuses as much of `DDPG` as it can: its `Actor`/`EnsembleQ` network
    classes (`..ddpg.models`, unchanged -- the extra time/match features are
    simply concatenated onto the state tensor before every forward call,
    exactly as the reference's own hand-written `Actor.forward(self, x, t)`
    does), its noise utilities, its `soft_update`, and this repo's generic
    `ReplayBuffer`/`NStepReplay` (both already iterate generically over
    whatever obs keys are given, so storing "obs", "t_to_horizon", and
    optionally "match" as separate obs-dict keys costs nothing extra there).

    Known, disclosed simplifications relative to the reference:
      - The reference stages a just-finished episode's transitions in a
        temporary buffer and only pushes them into the real replay buffer
        once the episode's OT reward is known, because it runs one
        environment sequentially and processes episodes as they arrive. This
        repo runs many synchronized, fixed-length parallel environments, so
        `explore_env` simply collects one full episode across *all* of them
        per call (`timesteps == self.horizon`, enforced, where `self.horizon`
        is the expert demos' own trajectory length -- see `__init__`) and
        computes every env's OT reward at once afterward -- no staging buffer
        needed.
      - The reference bootstraps its policy-input normalizer from a mix of
        early random-exploration data and the expert demonstrations, with a
        floor on the standard deviation. This class instead uses this repo's
        standard running `RunningMeanStd` (as every other agent here does)
        for policy-input normalization; only the OT reward's own atom
        normalizer (`OOPSRewarder`) is fit once from the expert demos, as in
        the reference.
    """

    def __init__(self, full_cfg, logdir=None, **kwargs):
        self.network_config = full_cfg.agent.network
        self.ddpg_config = full_cfg.agent.ddpg
        self.num_actors = self.ddpg_config.num_actors
        self.max_agent_steps = int(self.ddpg_config.max_agent_steps)
        super().__init__(full_cfg, logdir=logdir, **kwargs)

        self.oops_config = full_cfg.agent.get("oops", {})
        self.state_action = bool(self.oops_config.get("state_action", False))
        self.use_match = bool(self.oops_config.get("use_match", False))
        self.aug_time = bool(self.oops_config.get("aug_time", True))
        self.expl_noise = float(self.oops_config.get("expl_noise", 0.2))
        self.policy_noise = float(self.oops_config.get("policy_noise", 0.1))
        self.noise_clip = float(self.oops_config.get("noise_clip", 0.5))
        self.actor_clip = float(self.oops_config.get("actor_clip", 0.0))
        self.critic_clip = float(self.oops_config.get("critic_clip", 25.0))
        self.pal_alpha = float(self.oops_config.get("pal_alpha", 0.4))
        self.pal_min_priority = float(self.oops_config.get("pal_min_priority", 1.0))

        # --- Normalizers (policy input only; the OT reward has its own,
        # fixed, expert-fit normalizer inside OOPSRewarder) ---
        rms_config = {"eps": 1e-4, "with_clamp": True, "initial_count": "eps"}
        if self.normalize_input:
            self.obs_rms = {}
            for k, v in self.obs_space.items():
                if re.match(self.obs_rms_keys, k):
                    self.obs_rms[k] = normalizers.RunningMeanStd(v, **rms_config)
                else:
                    self.obs_rms[k] = normalizers.Identity()
            self.obs_rms = nn.ModuleDict(self.obs_rms).to(self.device)
        else:
            self.obs_rms = None

        # --- Demos + OT rewarder ---
        demos_config = self.oops_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)
        # The demos are the fixed artifact here: OOPSRewarder needs the agent's
        # collected episode and the expert trajectory to be the exact same
        # length (Sinkhorn's cost matrix must be square). So this defaults to
        # the DEMOS' own (already-subsampled) trajectory length, not
        # `env.max_episode_length` -- those can disagree (e.g. some of this
        # repo's DFlex demo files were recorded with fewer steps than that
        # env's current default episode length), and `env.max_episode_length`
        # would silently be the wrong number to match `ddpg.horizon_len`
        # against.
        configured_horizon = self.oops_config.get("time_horizon", None)
        self.horizon = int(configured_horizon) if configured_horizon else int(self.demos["act"].shape[1])
        self.oops_rewarder = OOPSRewarder(
            self.demos,
            device=self.device,
            state_action=self.state_action,
            reward_scale=float(self.oops_config.get("reward_scale", 5.0)),
            sinkhorn_eps=float(self.oops_config.get("sinkhorn_eps", 0.05)),
            sinkhorn_iters=int(self.oops_config.get("sinkhorn_iters", 1000)),
            std_clip=float(self.oops_config.get("std_clip", 0.33)),
            clip_state=bool(self.oops_config.get("clip_state", True)),
            time_horizon=self.horizon,
        )
        self._last_oops_stats = {}

        # --- Model: state_dim inflated by the time feature (+ match, if
        # enabled) relative to plain DDPG, since the actor/critic take those
        # concatenated onto the (normalized) observation. ---
        obs_dim = self.obs_space["obs"]
        obs_dim = obs_dim[0] if isinstance(obs_dim, tuple) else obs_dim
        self.obs_dim = obs_dim
        extra_dims = 1 + int(self.use_match)
        actor_state_dim = obs_dim + 1  # actor never sees "match" (see class docstring)
        critic_state_dim = obs_dim + extra_dims

        ActorCls = getattr(models, self.network_config.actor)
        CriticCls = getattr(models, self.network_config.critic)
        self.actor = ActorCls(actor_state_dim, self.action_dim, **self.network_config.get("actor_kwargs", {}))
        self.critic = CriticCls(critic_state_dim, self.action_dim, **self.network_config.get("critic_kwargs", {}))
        self.actor.to(self.device)
        self.critic.to(self.device)
        print("Actor:", self.actor)
        print("Critic:", self.critic, "\n")

        OptimCls = getattr(torch.optim, self.ddpg_config.optim_type)
        self.actor_optim = OptimCls(self.actor.parameters(), **self.ddpg_config.get("actor_optim_kwargs", {}))
        self.critic_optim = OptimCls(self.critic.parameters(), **self.ddpg_config.get("critic_optim_kwargs", {}))

        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor) if not self.ddpg_config.no_tgt_actor else self.actor

        # --- Buffers: "t_to_horizon" (and optionally "match") are stored as
        # ordinary extra obs-dict keys, so the generic ReplayBuffer/NStepReplay
        # need no changes at all to carry them alongside "obs". ---
        self.oops_obs_space = dict(self.obs_space)
        self.oops_obs_space["t_to_horizon"] = (1,)
        if self.use_match:
            self.oops_obs_space["match"] = (1,)
        self.memory = ReplayBuffer(
            self.oops_obs_space, self.action_dim, capacity=int(self.ddpg_config.memory_size), device=self.device
        )
        self.n_step_buffer = NStepReplay(
            self.oops_obs_space, self.action_dim, self.num_actors, self.ddpg_config.nstep, device=self.device
        )
        self.reward_shaper = RewardShaper(**self.ddpg_config.get("reward_shaper", {"fn": "scale", "scale": 1.0}))

        self.timer_total_names = ("agent.explore_env", "memory.add_to_buffer", "agent.update_net")

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def _augmented_state(self, obs, t_to_horizon, match=None):
        if self.normalize_input:
            x = self.obs_rms["obs"].normalize(obs["obs"])
        else:
            x = obs["obs"]
        parts = [x, t_to_horizon]
        if match is not None:
            parts.append(match)
        return torch.cat(parts, dim=-1)

    def get_actions(self, obs, t_to_horizon, sample=True):
        x = self._augmented_state(obs, t_to_horizon)
        mu, _, _ = self.actor(x)
        actions = mu
        if sample:
            actions = add_normal_noise(actions, std=self.expl_noise, out_bounds=[-1.0, 1.0])
        return actions

    @torch.no_grad()
    def _target_policy_actions(self, next_obs, next_t_to_horizon):
        x = self._augmented_state(next_obs, next_t_to_horizon)
        mu, _, _ = self.actor_target(x)
        return add_normal_noise(mu, std=self.policy_noise, noise_bounds=[-self.noise_clip, self.noise_clip], out_bounds=[-1.0, 1.0])

    # ------------------------------------------------------------------
    # Rollout collection: one full episode per call, reward assigned once
    # the episode (and therefore its OT matching) is complete.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        if timesteps != self.horizon:
            raise ValueError(
                "OOPS needs warm_up == horizon_len == the demos'/env's episode length: "
                "its reward is only defined for a whole completed episode, matched via "
                "Sinkhorn OT against one full expert trajectory of the same length."
            )

        episode_obs = torch.empty((self.num_actors, timesteps, self.obs_dim), device=self.device)
        second_dim = self.action_dim if self.state_action else self.obs_dim
        episode_second = torch.empty((self.num_actors, timesteps, second_dim), device=self.device)

        traj_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.oops_obs_space.items()
        }
        traj_actions = torch.empty((self.num_actors, timesteps, self.action_dim), device=self.device)
        traj_next_obs = {
            k: torch.empty((self.num_actors, timesteps) + v, dtype=torch.float32, device=self.device)
            for k, v in self.oops_obs_space.items()
        }
        traj_dones = torch.empty((self.num_actors, timesteps), device=self.device)
        env_rewards_sum = torch.zeros(self.num_actors, device=self.device)
        raw_env_rewards_sum = torch.zeros_like(env_rewards_sum)

        for i in range(timesteps):
            if not self.env_autoresets:
                raise NotImplementedError

            if self.normalize_input:
                self.obs_rms["obs"].update(self.obs["obs"])

            t_to_horizon = torch.full((self.num_actors, 1), (timesteps - i) / float(timesteps), device=self.device)

            if random:
                actions = torch.rand((self.num_actors, self.action_dim), device=self.device) * 2.0 - 1.0
            else:
                actions = self.get_actions(self.obs, t_to_horizon, sample=sample)

            next_obs, rewards, dones, infos = env.step(actions)
            terminal_obs = self._convert_obs(infos.get('obs_before_reset', next_obs))
            next_obs = self._convert_obs(next_obs)
            raw_env_rewards_sum += rewards
            rewards = self._reported_rewards(rewards, dones, infos)
            env_rewards_sum += rewards

            done_indices = torch.where(dones)[0].tolist()
            if done_indices and i < timesteps - 1:
                raise RuntimeError(
                    f"OOPS environment reset before the fixed horizon at step {i + 1}/{timesteps} "
                    f"(envs {done_indices}). Disable early termination and check simulation stability; "
                    "OT matching cannot use a trajectory containing an automatic reset."
                )
            self.metrics.update(self.epoch, self.env, self.obs, rewards, done_indices, infos)

            if self.ddpg_config.handle_timeout:
                dones = self._handle_timeout(dones, infos)

            episode_obs[:, i] = self.obs["obs"]
            episode_second[:, i] = actions if self.state_action else terminal_obs["obs"]

            traj_obs["obs"][:, i] = self.obs["obs"]
            traj_obs["t_to_horizon"][:, i] = t_to_horizon
            traj_actions[:, i] = actions
            traj_dones[:, i] = dones
            traj_next_obs["obs"][:, i] = terminal_obs["obs"]
            next_t_to_horizon = torch.full(
                (self.num_actors, 1), max(timesteps - i - 1, 0) / float(timesteps), device=self.device
            )
            traj_next_obs["t_to_horizon"][:, i] = next_t_to_horizon
            self.obs = next_obs

        self.metrics.flush_video(self.epoch)

        demo_indices = self.oops_rewarder.sample_demo_indices(self.num_actors)
        ot_rewards, matching = self.oops_rewarder.compute_episode_reward(episode_obs, episode_second, demo_indices)
        traj_rewards = ot_rewards

        if self.use_match:
            # Matches the reference's own (unclamped) `match/1000.` and
            # `(match+1.)/1000.`; both are already within [0, 1] by
            # construction since `matching` ranges over [0, horizon - 1].
            match_normed = matching.to(torch.float32) / float(self.horizon)
            next_match_normed = (matching.to(torch.float32) + 1.0) / float(self.horizon)
            traj_obs["match"] = match_normed.unsqueeze(-1)
            traj_next_obs["match"] = next_match_normed.unsqueeze(-1)

        self._last_oops_stats = {
            "reward_mean": traj_rewards.mean(),
            "reward_std": traj_rewards.std(unbiased=False),
            "env_return_mean": env_rewards_sum.mean(),
            "raw_env_return_mean": raw_env_rewards_sum.mean(),
            "expert_return": torch.tensor(float(self.demos["expert_return"]), device=self.device),
        }

        traj_rewards = self.reward_shaper(traj_rewards.reshape(self.num_actors, timesteps, 1))
        traj_dones = traj_dones.reshape(self.num_actors, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones)

        return data, timesteps * self.num_actors

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.ones((self.num_actors,), dtype=torch.bool, device=self.device)

        self.set_eval()
        trajectory, steps = self.explore_env(self.env, self.ddpg_config.warm_up, random=True)
        self.memory.add_to_buffer(trajectory)
        self.agent_steps += steps

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            if self.job_clock is not None:
                _, safe_stop = self.job_clock.step(check_safe_stop=True)
                if safe_stop:
                    print("Not enough time left for another step. Exiting cleanly.")
                    break
            self.set_eval()
            trajectory, steps = self.explore_env(self.env, self.ddpg_config.horizon_len, sample=True)
            self.agent_steps += steps
            self.memory.add_to_buffer(trajectory)

            self.set_train()
            results = self.update_net(self.memory)

            # Some entries (e.g. "grad_norm/actor") are None whenever the
            # corresponding clip value is disabled (e.g. actor_clip == 0, the
            # paper's own default) -- drop those rather than stacking them.
            metrics = {
                k: torch.mean(torch.stack(values)).item()
                for k, v in results.items()
                if (values := [x for x in v if x is not None])
            }
            metrics.update({"epoch": self.epoch, "mini_epoch": self.mini_epoch})
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}
            metrics.update({f"oops/{k}": v.item() for k, v in self._last_oops_stats.items()})

            episode_metrics = {
                "train_scores/episode_rewards": self.metrics.episode_trackers["rewards"].mean(),
                "train_scores/episode_lengths": self.metrics.episode_trackers["lengths"].mean(),
                "train_scores/num_episodes": self.metrics.num_episodes,
                **self.metrics.result(prefix="train"),
            }
            metrics.update(episode_metrics)

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            self._checkpoint_save(episode_metrics["train_scores/episode_rewards"])

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(
                    f"Epochs: {self.epoch + 1} |",
                    f"Agent Steps: {int(self.agent_steps):,} |",
                    f"ep_rewards {episode_metrics['train_scores/episode_rewards']:.2f} |",
                    f"Expert return: {self.demos['expert_return']:.2f}",
                )

        self.save(os.path.join(self.ckpt_dir, "final.pth"))

    @staticmethod
    def _reported_rewards(rewards, dones, infos):
        # Reference main.py zeros reward on each done step, but keeps stepping.
        terminal = infos.get('termination', dones).bool() | dones.bool()
        return rewards.masked_fill(terminal, 0.0)

    def _updates_per_rollout(self):
        configured = self.ddpg_config.mini_epochs
        return self.num_actors * self.horizon if configured is None else int(configured)

    def update_net(self, memory):
        results = collections.defaultdict(list)
        for _i in range(self._updates_per_rollout()):
            self.mini_epoch += 1
            obs, action, reward, next_obs, done = memory.sample_batch(self.ddpg_config.batch_size, device=self.device)

            t_to_horizon = obs["t_to_horizon"]
            next_t_to_horizon = next_obs["t_to_horizon"]
            match = obs.get("match")
            next_match = next_obs.get("match")
            # The actor's own policy-gradient step always conditions on the
            # transition's real, recorded time-to-horizon (matching the
            # reference's `t_H_normed = t/1000.` fed to `self.actor(...)` in
            # its actor-loss line) -- only the critic's TD regression below
            # trains against the (optionally time-augmented) resampled value.
            actor_t_to_horizon = t_to_horizon

            if self.aug_time:
                # Data augmentation over the time feature: retrain the (time,
                # state, action)-conditioned value/policy on a uniformly
                # resampled time-to-horizon, decoupled from what this
                # particular transition actually recorded, following the
                # reference's own `--aug_time` option.
                t_H_plus = torch.randint(1, self.horizon + 1, t_to_horizon.shape, device=self.device).float()
                t_to_horizon = t_H_plus / self.horizon
                next_t_to_horizon = (t_H_plus - 1.0).clamp(min=0.0) / self.horizon

            # Raw observations: `_augmented_state` (used by acting, the critic and
            # the actor alike) applies the running normalizer itself. Normalizing
            # here as well used to normalize twice during training but only once
            # when acting, so the networks trained on a different input
            # distribution than the one the policy actually sees.
            obs_n = {"obs": obs["obs"]}
            next_obs_n = {"obs": next_obs["obs"]}

            critic_loss, critic_grad_norm = self.update_critic(
                obs_n, action, reward, next_obs_n, done, t_to_horizon, next_t_to_horizon, match, next_match
            )
            results["loss/critic"].append(critic_loss.detach())
            results["grad_norm/critic"].append(critic_grad_norm)

            if self.mini_epoch % self.ddpg_config.update_actor_interval == 0:
                actor_loss, actor_grad_norm = self.update_actor(obs_n, actor_t_to_horizon, match)
                results["loss/actor"].append(actor_loss.detach())
                results["grad_norm/actor"].append(actor_grad_norm)

            if self.mini_epoch % self.ddpg_config.update_targets_interval == 0:
                soft_update(self.critic_target, self.critic, self.ddpg_config.tau)
                if not self.ddpg_config.no_tgt_actor:
                    soft_update(self.actor_target, self.actor, self.ddpg_config.tau)
        return results

    def update_critic(self, obs, action, reward, next_obs, done, t_to_horizon, next_t_to_horizon, match, next_match):
        with torch.no_grad():
            next_actions = self._target_policy_actions(next_obs, next_t_to_horizon)
            next_x = self._augmented_state(next_obs, next_t_to_horizon, next_match)
            target_Qs = self.critic_target.get_q_values(next_x, next_actions)
            target_Q = torch.min(torch.stack(target_Qs), dim=0).values
            # Reference TD3.py uses t_H_plus > 1, including for augmented time.
            # Environment timeouts may be cleared in replay and are not this mask.
            not_done = (next_t_to_horizon > 0).to(reward.dtype).reshape_as(reward)
            target_Q = reward + not_done * (self.ddpg_config.gamma**self.ddpg_config.nstep) * target_Q

        x = self._augmented_state(obs, t_to_horizon, match)
        current_Qs = self.critic.get_q_values(x, action)

        td_losses = [current_Q - target_Q for current_Q in current_Qs]
        critic_loss = sum(self._pal(td) for td in td_losses)
        priority_norm = torch.stack([td.abs() for td in td_losses]).max(dim=0).values
        priority_norm = priority_norm.clamp(min=self.pal_min_priority).pow(self.pal_alpha).mean().detach()
        critic_loss = critic_loss / priority_norm

        grad_norm = self._optimizer_update(self.critic_optim, critic_loss, self.critic, self.critic_clip)
        return critic_loss, grad_norm

    def update_actor(self, obs, t_to_horizon, match):
        self.critic.requires_grad_(False)
        x = self._augmented_state(obs, t_to_horizon)
        mu, _, _ = self.actor(x)
        critic_x = self._augmented_state(obs, t_to_horizon, match)
        # TD3's actor loss uses only the first critic ("Q1"), not the min of
        # both, matching the reference's own `self.critic.Q1(...)` call.
        Q = self.critic.get_q_values(critic_x, mu)[0]
        actor_loss = -Q.mean()
        grad_norm = self._optimizer_update(self.actor_optim, actor_loss, self.actor, self.actor_clip)
        self.critic.requires_grad_(True)
        return actor_loss, grad_norm

    def _pal(self, x):
        # Loss-Adjusted / Prioritized Approximation Loss (PAL), from LAP/PAL
        # (Fujimoto, Meger, Precup); if `pal_min_priority == 1`, this reduces
        # to plain Huber. Ported unchanged from the reference's `TD3.PAL`.
        return torch.where(
            x.abs() < self.pal_min_priority,
            (self.pal_min_priority**self.pal_alpha) * 0.5 * x.pow(2),
            self.pal_min_priority * x.abs().pow(1.0 + self.pal_alpha) / (1.0 + self.pal_alpha),
        ).mean()

    def _optimizer_update(self, optimizer, objective, module, clip_value):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        grad_norm = None
        if clip_value is not None and clip_value > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(module.parameters(), clip_value)
        optimizer.step()
        return grad_norm

    def eval(self):
        self.set_eval()

        obs = self.env.reset()
        self.obs = self._convert_obs(obs)

        total_eval_episodes = self.num_actors * 2
        eval_metrics = self._create_metrics(total_eval_episodes, self.metrics_kwargs)
        with self._as_metrics(eval_metrics), torch.no_grad():
            while self.metrics.num_episodes < total_eval_episodes:
                # One full episode per block, exactly like `explore_env`
                # (OOPS's actor is conditioned on time-to-horizon, so it must
                # be fed a full, correctly-numbered episode at a time).
                for i in range(self.horizon):
                    if not self.env_autoresets:
                        raise NotImplementedError

                    t_to_horizon = torch.full(
                        (self.num_actors, 1), (self.horizon - i) / float(self.horizon), device=self.device
                    )
                    actions = self.get_actions(self.obs, t_to_horizon, sample=True)

                    next_obs, rewards, dones, infos = self.env.step(actions)
                    next_obs = self._convert_obs(next_obs)
                    rewards = self._reported_rewards(rewards, dones, infos)

                    done_indices = torch.where(dones)[0].tolist()
                    self.metrics.update(self.epoch, self.env, self.obs, rewards, done_indices, infos)

                    self.obs = next_obs
            self.metrics.flush_video(self.epoch)

            metrics = {
                "eval_scores/num_episodes": self.metrics.num_episodes,
                "eval_scores/episode_rewards": self.metrics.episode_trackers["rewards"].mean(),
                "eval_scores/episode_lengths": self.metrics.episode_trackers["lengths"].mean(),
                **self.metrics.result(prefix="eval"),
            }
            print(metrics)

            self.writer.add(self.agent_steps, metrics)

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def save(self, f):
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        if self.normalize_input:
            state["obs_rms"] = self.obs_rms.state_dict()
        torch.save(state, f)

    def load(self, f):
        checkpoint = torch.load(f, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if self.normalize_input and "obs_rms" in checkpoint:
            self.obs_rms.load_state_dict(checkpoint["obs_rms"])
