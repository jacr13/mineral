import json
import os
import re

import torch

from ..ddpg.ddpg import DDPG
from .common import ExpertTransitions, slice_obs, sliced_space, state_dims
from .icvf import load_or_pretrain_icvf
from .models import Discriminator


class _Relabelled:
    """Replay view whose sampled rewards are recomputed by ``reward_fn(obs, actions, next_obs)``."""

    def __init__(self, memory, reward_fn):
        self.memory, self.reward_fn = memory, reward_fn

    def sample_batch(self, batch_size, device='cuda'):
        obs, action, _, next_obs, done = self.memory.sample_batch(batch_size, device=device)
        return obs, action, self.reward_fn(obs, action, next_obs), next_obs, done

    def __getattr__(self, name):
        return getattr(self.memory, name)


class LWAIL(DDPG):
    """Latent Wasserstein Adversarial Imitation Learning, built on this repo's DDPG.

    Reference: Yang, Yan, Schwing, Wang, "Latent Wasserstein Adversarial
    Imitation Learning" (ICLR 2026), https://arxiv.org/abs/2603.05440
    Official code: https://github.com/JackyYang258/LWAIL (see ``core.py``,
    ``network.py``, ``utils.py``).

    Upstream's ``core.py::Agent`` is downstream-RL-agnostic (``--downstream``
    selects TD3/DDPG/SAC/PPO), and its own published runs (``run.bash``,
    ``all_run.bash``) all use ``--downstream td3``. This repo has no separate
    TD3 class, but ``DDPG`` already implements TD3 as a configuration (twin
    critics via ``EnsembleQ``/``get_q_min``, delayed policy updates via
    ``update_actor_interval``, target policy smoothing via
    ``get_tgt_policy_actions``) -- see ``mineral/cfgs/agent/DDPG/DFlexAnt.yaml``,
    which is already set up this way. So LWAIL is built directly on ``DDPG``
    here, matching upstream's actual downstream choice.

    On top of that, LWAIL swaps the usual binary-classification discriminator
    for a 1-Lipschitz Wasserstein critic ("f_net" upstream): trained with a
    tanh-bounded IPM objective plus a WGAN-GP gradient penalty (``upstream:
    core.py::Agent.f_update/pretrain``, ``utils.py::gradient_penalty``)
    instead of BCE, and it shapes reward as ``sigmoid(-f_net(...))`` instead of
    ``-log(1 - sigmoid(f_net(...)))``. The discriminator network itself and its
    ``state``/``state_action``/``state_state`` input modes reuse
    :class:`~mineral.agents.il.models.Discriminator`, the same class GAIL/DAC/
    OPOLO use.

    Two more upstream options are ported as-is:
      - ``lwail.minus``: for ``input_type: state_state``, feed the critic
        ``next_obs - obs`` instead of raw ``next_obs`` (upstream ``--minus``).
      - ``lwail.using_icvf``: embed states through a frozen ICVF encoder
        (:class:`~mineral.agents.il.icvf_models.PhiNet`) before feeding the
        critic (upstream ``--using_icvf``). Upstream only ships ICVF
        checkpoints for a few D4RL Mujoco tasks (dimension-incompatible with
        this repo's dflex tasks) and points to a separate, unfinished
        "ICVF-PyTorch Repository (todo)" for the training code; see
        ``icvf.py`` for a from-scratch port of that training code (of the
        official JAX ICVF release). Nothing extra to run: the first time a
        given ``lwail.icvf_path`` doesn't exist (default:
        ``icvf_model/dflex_<env_name>.pt``), ``__init__`` pretrains one from a
        random rollout of ``self.env`` and caches it there (see
        ``icvf.py::load_or_pretrain_icvf``); every later run for that path
        (other seeds, sweeps) just loads the cached checkpoint. On by default
        (``LWAIL/DFlexAnt.yaml``): without it, the raw-state critic saturates
        and the reward signal collapses on higher-dimensional envs; set
        ``lwail.using_icvf: false`` to disable.

    Upstream also pretrains "f_net" for ``lwail.pretrain_iters`` steps (default
    2500, matching upstream's hardcoded value) against the warm-up random
    rollout before online training starts (``upstream:
    core.py::Agent.pretrain``); see ``_pretrain_discriminator`` below.
    """

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)

        self.lwail_config = full_cfg.agent.get("lwail", {})
        self.input_type = self.lwail_config.get("input_type", "state_state")

        assert int(self.ddpg_config.nstep) == 1, "LWAIL scores 1-step (s, s') pairs: set agent.ddpg.nstep=1"
        self.expert = ExpertTransitions(self.device, self.lwail_config.demos)
        self.demos = {"expert_return": self.expert.expert_return}

        discriminator_config = self.lwail_config.get("discriminator", {})
        encoder_kwargs = discriminator_config.get("encoder_kwargs", None)
        if encoder_kwargs is None:
            encoder_kwargs = self.network_config.get("encoder_kwargs", {})

        self.state_d = state_dims(full_cfg, self.obs_space, self.lwail_config.get("state_obs", False))
        self.discriminator = Discriminator(
            sliced_space(self.obs_space, self.state_d),
            self.action_dim,
            input_type=self.input_type,
            discriminator_kwargs=discriminator_config,
            encoder_kwargs=encoder_kwargs,
        ).to(self.device)
        disc_optim_kwargs = discriminator_config.get("optim", {"type": "Adam", "kwargs": {"lr": 1e-3}})
        DiscOptim = getattr(torch.optim, disc_optim_kwargs.get("type", "Adam"))
        self.disc_optim = DiscOptim(self.discriminator.parameters(), **disc_optim_kwargs.get("kwargs", {}))

        self.disc_iters = int(discriminator_config.get("iters", 30))  # upstream `--f_epoch`
        self.expert_batch_size = int(self.lwail_config.get("expert_batch_size", self.ddpg_config.batch_size))
        self.policy_batch_size = int(self.lwail_config.get("policy_batch_size", self.ddpg_config.batch_size))
        self.reward_scale = float(self.lwail_config.get("reward_scale", 1.0))
        self.disc_update_freq = int(self.lwail_config.get("disc_update_freq", 1))

        self.gp_coef = float(self.lwail_config.get("gp_coef", 10.0))  # upstream `--alpha`
        self.minus = bool(self.lwail_config.get("minus", True))
        self.pretrain_iters = int(self.lwail_config.get("pretrain_iters", 2500))

        self.using_icvf = bool(self.lwail_config.get("using_icvf", False))
        self.phi_net = None
        if self.using_icvf:
            icvf_hidden_dims = list(self.lwail_config.get("icvf_hidden_dims", [256, 256]))
            env_name = full_cfg.task.env.get("env_name", full_cfg.task.name)
            suffix = f"_s{self.state_d}" if self.state_d else ""
            icvf_path = self.lwail_config.get("icvf_path", None) or f"icvf_model/dflex_{env_name}{suffix}.pt"

            obs_dim = self.state_d or self.obs_space["obs"][0]
            icvf_pretrain_config = dict(self.lwail_config.get("icvf_pretrain", {}))
            self.phi_net = load_or_pretrain_icvf(
                self.env,
                obs_dim,
                icvf_hidden_dims,
                icvf_path,
                self.device,
                pretrain_kwargs=icvf_pretrain_config,
                state_dims=self.state_d,
            )

            # Rebuild the discriminator (constructed above against the raw obs
            # space) to instead operate on the ICVF-embedded state space.
            embed_dim = icvf_hidden_dims[-1]
            disc_obs_space = {"obs": (embed_dim,)}
            self.discriminator = Discriminator(
                disc_obs_space,
                self.action_dim,
                input_type=self.input_type,
                discriminator_kwargs=discriminator_config,
                encoder_kwargs=encoder_kwargs,
            ).to(self.device)
            self.disc_optim = DiscOptim(self.discriminator.parameters(), **disc_optim_kwargs.get("kwargs", {}))

        self.discriminator.eval()

    def _normalize_obs_dict(self, obs):
        if self.normalize_input:
            return {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        return obs

    def get_discriminator_inputs(self, obs, actions, next_obs):
        """Critic inputs from RAW observations.

        The frozen ICVF encoder (trained on raw states) when enabled, otherwise the agent's observation normaliser;
        the same transform for expert and agent data.
        """
        if self.using_icvf:  # ICVF trained on the raw state part
            obs = {"obs": self.phi_net(slice_obs(obs, self.state_d)["obs"])}
            next_obs = {"obs": self.phi_net(slice_obs(next_obs, self.state_d)["obs"])}
        else:  # normalise the whole vector, keep its state part
            obs = slice_obs(self._normalize_obs_dict(obs), self.state_d)
            next_obs = slice_obs(self._normalize_obs_dict(next_obs), self.state_d)

        if self.input_type == "state":
            return obs, None
        if self.input_type == "state_action":
            return obs, actions.clamp(-1, 1)
        if self.input_type == "state_state":
            next_obs_in = {"obs": next_obs["obs"] - obs["obs"]} if self.minus else next_obs
            return obs, next_obs_in
        raise NotImplementedError

    @torch.no_grad()
    def lwail_reward(self, obs, actions, next_obs):
        # LWAIL reward: sigmoid(-f_net(...)), vs. GAIL/DAC's -log(1 - sigmoid(f_net(...))).
        input_1, input_2 = self.get_discriminator_inputs(obs, actions, next_obs)
        logits = self.discriminator(input_1, input_2)
        reward = torch.sigmoid(-logits)
        return (reward * self.reward_scale).unsqueeze(-1)

    @staticmethod
    def _leaf_tensors(x):
        if x is None:
            return []
        if isinstance(x, dict):
            return list(x.values())
        return [x]

    @staticmethod
    def _interp(a, b, eps):
        if a is None:
            return None
        if isinstance(a, dict):
            out = {}
            for k in a:
                v = (eps * a[k] + (1.0 - eps) * b[k]).detach()
                v.requires_grad_(True)
                out[k] = v
            return out
        v = (eps * a + (1.0 - eps) * b).detach()
        v.requires_grad_(True)
        return v

    def _gradient_penalty(self, exp_input_1, exp_input_2, pol_input_1, pol_input_2):
        # Ported from upstream `utils.py::gradient_penalty`, adapted to the
        # dict/tensor discriminator inputs used throughout this repo: interpolate
        # each raw component separately with a shared per-sample coefficient
        # (equivalent to interpolating the concatenated vector upstream does),
        # then penalize the critic's input gradient norm for deviating from 1.
        batch_size = exp_input_1["obs"].shape[0]
        eps = torch.rand(batch_size, 1, device=self.device)

        interp_1 = self._interp(exp_input_1, pol_input_1, eps)
        interp_2 = self._interp(exp_input_2, pol_input_2, eps)

        logits = self.discriminator(interp_1, interp_2)
        inputs = self._leaf_tensors(interp_1) + self._leaf_tensors(interp_2)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=inputs,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        grad_sq = sum((g.reshape(g.shape[0], -1) ** 2).sum(dim=1) for g in grads)
        grad_norm = torch.sqrt(grad_sq + 1e-12)
        return ((grad_norm - 1.0) ** 2).mean() * self.gp_coef

    def _disc_step(self, exp_input_1, exp_input_2, pol_input_1, pol_input_2):
        # Ported from upstream `core.py::Agent.f_update`'s inner loop: a tanh-bounded
        # IPM (mean critic value on expert vs. policy data) plus a WGAN-GP penalty.
        exp_logits = self.discriminator(exp_input_1, exp_input_2)
        pol_logits = self.discriminator(pol_input_1, pol_input_2)
        ipm = torch.tanh(exp_logits).mean() - torch.tanh(pol_logits).mean()
        gp = self._gradient_penalty(exp_input_1, exp_input_2, pol_input_1, pol_input_2)
        loss = ipm + gp

        self.disc_optim.zero_grad()
        loss.backward()
        self.disc_optim.step()
        return loss.detach(), ipm.detach(), gp.detach()

    def train_discriminator(self, memory):

        self.discriminator.train()

        # Matches upstream: both the policy and expert batches are sampled once
        # per call, then `disc_iters` (upstream `f_epoch`) gradient steps are
        # taken against that same fixed pair (only the GP interpolation is
        # re-randomized every step).
        pol_obs, pol_act, _, pol_next_obs, _ = memory.sample_batch(self.policy_batch_size, device=self.device)
        exp_obs, exp_act, exp_next_obs, _ = self.expert.sample(self.expert_batch_size)

        pol_input_1, pol_input_2 = self.get_discriminator_inputs(pol_obs, pol_act, pol_next_obs)
        exp_input_1, exp_input_2 = self.get_discriminator_inputs(exp_obs, exp_act, exp_next_obs)

        losses, ipms, gps = [], [], []
        for _ in range(self.disc_iters):
            loss, ipm, gp = self._disc_step(exp_input_1, exp_input_2, pol_input_1, pol_input_2)
            losses.append(loss)
            ipms.append(ipm)
            gps.append(gp)

        self.discriminator.eval()
        return {
            "discriminator/total": torch.stack(losses).mean().item(),
            "discriminator/ipm": torch.stack(ipms).mean().item(),
            "discriminator/gp": torch.stack(gps).mean().item(),
        }

    def _pretrain_discriminator(self):
        # Ported from upstream `core.py::Agent.pretrain`: warm-start the critic
        # against the warm-up random rollout before online training starts. The
        # policy-side batch is sampled once and held fixed (as upstream does),
        # while the expert batch is refreshed every iteration (also as upstream
        # does, unlike the periodic `train_discriminator` updates above).
        if self.pretrain_iters <= 0:
            return

        print(f"[LWAIL] Pretraining discriminator for {self.pretrain_iters} iters on warm-up rollout vs. expert demos...")
        self.discriminator.train()

        pol_obs, pol_act, _, pol_next_obs, _ = self.memory.sample_batch(self.policy_batch_size, device=self.device)
        pol_input_1, pol_input_2 = self.get_discriminator_inputs(pol_obs, pol_act, pol_next_obs)

        losses = []
        for _ in range(self.pretrain_iters):
            exp_obs, exp_act, exp_next_obs, _ = self.expert.sample(self.expert_batch_size)
            exp_input_1, exp_input_2 = self.get_discriminator_inputs(exp_obs, exp_act, exp_next_obs)
            loss, _, _ = self._disc_step(exp_input_1, exp_input_2, pol_input_1, pol_input_2)
            losses.append(loss)

        self.discriminator.eval()
        metrics = {"train_stats/pretrain/discriminator_loss": torch.stack(losses).mean().item()}
        self.writer.add(self.agent_steps, metrics)
        self.writer.write()

    def update_net(self, memory):
        """DDPG/TD3 update with the reward of every sampled batch relabelled by the current critic f."""
        self.discriminator.eval()
        return super().update_net(_Relabelled(memory, self.lwail_reward))

    def train(self):
        # Identical to DDPG.train() except for: the discriminator pretrain
        # phase after warm-up, the periodic discriminator update, and richer
        # logging (expert_return, discriminator metrics, best-checkpoint
        # tracking, and a print_every summary), matching GAIL/DAC's loops.
        if not self._resumed:  # a resumed run has its env state, replay buffer and counters from the checkpoint
            obs = self.env.reset()
            self.obs = self._convert_obs(obs)
            self.dones = torch.ones((self.num_actors,), dtype=torch.bool, device=self.device)

            self.set_eval()
            trajectory, steps = self.explore_env(self.env, self.ddpg_config.warm_up, random=True)
            self.memory.add_to_buffer(trajectory)
            self.agent_steps += steps
            self._pretrain_discriminator()

        self.protocol_begin()

        while self.agent_steps < self.max_agent_steps:
            self.protocol_step()
            self.epoch += 1
            if self.job_clock is not None:
                _, safe_stop = self.job_clock.step(check_safe_stop=True)
                if safe_stop:
                    print("Not enough time left for another step. Exiting cleanly.")
                    self.protocol_clock_stop()
                    break

            self.set_eval()
            trajectory, steps = self.explore_env(self.env, self.ddpg_config.horizon_len, sample=True)
            self.agent_steps += steps
            self.memory.add_to_buffer(trajectory)

            disc_metrics = {}
            if self.epoch % self.disc_update_freq == 0:
                disc_metrics = self.train_discriminator(self.memory)

            self.set_train()
            results = self.update_net(self.memory)

            metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
            metrics.update({"epoch": self.epoch, "mini_epoch": self.mini_epoch})
            metrics.update({"expert_return": self.demos["expert_return"]})
            metrics = {f"train_stats/{k}": v for k, v in metrics.items()}

            episode_metrics = {
                "train_scores/episode_rewards": self.metrics.episode_trackers["rewards"].mean(),
                "train_scores/episode_lengths": self.metrics.episode_trackers["lengths"].mean(),
                "train_scores/num_episodes": self.metrics.num_episodes,
                **self.metrics.result(prefix="train"),
                **disc_metrics,
            }
            metrics.update(episode_metrics)

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            self._checkpoint_save(metrics["train_scores/episode_rewards"])

            if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                print(
                    f"Epochs: {self.epoch + 1} |",
                    f"Agent Steps: {int(self.agent_steps):,} |",
                    f"Best: {self.best_stat if self.best_stat is not None else -float('inf'):.2f} |",
                    "Stats:",
                    f"ep_rewards {episode_metrics['train_scores/episode_rewards']:.2f},",
                    f"ep_lengths {episode_metrics['train_scores/episode_lengths']:.2f},",
                    f"Expert return: {self.demos['expert_return']:.2f}",
                )

        self.protocol_end(self.max_agent_steps)

        self.save(os.path.join(self.ckpt_dir, "final.pth"))

    def eval(self):
        # `DDPG.eval/save/load` are unimplemented stubs (this repo has no task
        # that runs plain `DDPG` to completion), so best-checkpoint tracking,
        # `run: train_eval`, and `--ckpt` resume all no-op silently on top of
        # `DDPG`. Implemented here, following `SAC.eval/save/load`'s pattern,
        # deterministically (`sample=False`) since LWAIL's downstream policy is
        # deterministic (DDPG/TD3), unlike SAC's stochastic eval.
        self.set_eval()

        obs = self.env.reset()
        obs = self._convert_obs(obs)

        total_eval_episodes = self.num_actors * 2
        eval_metrics = self._create_metrics(total_eval_episodes, self.metrics_kwargs)
        with self._as_metrics(eval_metrics), torch.no_grad():
            eval_counts = [0] * self.num_actors
            while min(eval_counts) < 2:
                if not self.env_autoresets:
                    raise NotImplementedError

                actions = self.get_actions(obs=obs, sample=False)
                next_obs, rewards, dones, infos = self.env.step(actions)
                next_obs = self._convert_obs(next_obs)
                rewards, dones = (
                    torch.as_tensor(rewards, device=self.device),
                    torch.as_tensor(dones, device=self.device),
                )

                done_indices = torch.where(dones)[0].tolist()
                done_indices = self._eval_keep(done_indices, eval_counts)
                self.metrics.update(self.epoch, self.env, obs, rewards, done_indices, infos)

                obs = next_obs
            self.metrics.flush_video(self.epoch)

            metrics = {
                "eval_scores/num_episodes": self.metrics.num_episodes,
                "eval_scores/episode_rewards": self.metrics.episode_trackers["rewards"].mean(),
                "eval_scores/episode_lengths": self.metrics.episode_trackers["lengths"].mean(),
                **self.metrics.result(prefix="eval"),
            }
            print(metrics)

            self.writer.add(self.agent_steps, metrics)
            self.writer.write()

            scores = {
                "epoch": self.epoch,
                "mini_epoch": self.mini_epoch,
                "agent_steps": self.agent_steps,
                "eval_scores/num_episodes": self.metrics.num_episodes,
                "eval_scores/episode_rewards": list(self.metrics.episode_trackers["rewards"].window),
                "eval_scores/episode_lengths": list(self.metrics.episode_trackers["lengths"].window),
            }
            json.dump(scores, open(os.path.join(self.logdir, "scores.json"), "w"), indent=4)

    def save(self, f):
        ckpt = {
            "epoch": self.epoch,
            "mini_epoch": self.mini_epoch,
            "agent_steps": self.agent_steps,
            "obs_rms": self.obs_rms.state_dict() if self.normalize_input else None,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict() if not self.ddpg_config.no_tgt_actor else None,
            "critic_target": self.critic_target.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }
        torch.save(ckpt, f)

    def load(self, f, ckpt_keys=""):
        all_ckpt_keys = ("epoch", "mini_epoch", "agent_steps")
        all_ckpt_keys += ("obs_rms", "actor", "critic", "actor_target", "critic_target", "discriminator")
        ckpt = torch.load(f, map_location=self.device)
        for k in all_ckpt_keys:
            if not re.match(ckpt_keys, k):
                print(f"Warning: ckpt skipped loading `{k}`")
                continue
            if k == "obs_rms" and (not self.normalize_input):
                continue
            if k == "actor_target" and self.ddpg_config.no_tgt_actor:
                continue

            if hasattr(getattr(self, k), "load_state_dict"):
                getattr(self, k).load_state_dict(ckpt[k])
            else:
                setattr(self, k, ckpt[k])
