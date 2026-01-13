import collections
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common.demos import get_demos
from ..ddpg.models import InverseModel
from ..ppo.ppo import PPO, actor_loss, bounds_loss, critic_loss, policy_kl
from ..ppo.utils import adjust_learning_rate_cos
from .models import Discriminator


class GAIL(PPO):
    """Generative Adversarial Imitation Learning built on PPO with minimal changes."""

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)

        # GAIL-specific config
        self.gail_config = full_cfg.agent.get("gail", {})

        self.input_type = self.gail_config.get("input_type", "state_action")

        # demos
        demos_config = self.gail_config.get("demos", {})
        self.demos = get_demos(self.device, **demos_config)

        # discriminator
        discriminator_config = self.gail_config.get("discriminator", {})
        encoder_kwargs = full_cfg.agent.network.get("encoder_kwargs", {})
        act_dim = self.action_dim
        self.discriminator = Discriminator(
            self.obs_space,
            act_dim,
            input_type=self.input_type,
            discriminator_kwargs=discriminator_config,
            encoder_kwargs=encoder_kwargs,
        ).to(self.device)

        disc_optim_kwargs = discriminator_config.get("optim", {"type": "Adam", "kwargs": {"lr": 3e-4}})
        DiscOptim = getattr(torch.optim, disc_optim_kwargs.get("type", "Adam"))
        self.disc_optim = DiscOptim(self.discriminator.parameters(), **disc_optim_kwargs.get("kwargs", {}))

        # training params
        self.disc_iters = int(discriminator_config.get("iters", 1))
        self.expert_batch_size = int(self.gail_config.get("expert_batch_size", self.minibatch_size))
        self.policy_batch_size = int(self.gail_config.get("policy_batch_size", self.minibatch_size))
        self.reward_scale = float(self.gail_config.get("reward_scale", 1.0))
        self.label_smooth = float(self.gail_config.get("label_smooth", 0.0))

        inv_config = self.gail_config.get("inverse_model", {})
        self.inv_reg_coef = float(inv_config.get("reg_coef", 0.0))
        self.inv_update_freq = int(inv_config.get("update_freq", 1))
        self.inv_batch_size = int(inv_config.get("batch_size", self.minibatch_size))
        self.inv_train_iters = int(inv_config.get("iters", 1))
        inv_patience = inv_config.get("patience", 0)
        self.inv_patience = 0 if inv_patience is None else int(inv_patience)
        self.inv_min_delta = float(inv_config.get("min_delta", 0.0))
        self.inverse_model = None
        self.inv_optim = None
        if inv_config.get("enabled", self.inv_reg_coef > 0.0):
            inv_mlp_kwargs = inv_config.get("mlp_kwargs", None)
            inv_encoder_kwargs = inv_config.get("encoder_kwargs", None)
            if inv_encoder_kwargs is None:
                inv_encoder_kwargs = full_cfg.agent.network.get("encoder_kwargs", {})
            self.inverse_model = InverseModel(
                obs_space=self.obs_space,
                action_dim=self.action_dim,
                mlp_kwargs=inv_mlp_kwargs,
                encoder_kwargs=inv_encoder_kwargs,
            ).to(self.device)
            inv_optim_kwargs = inv_config.get("optim", {"type": "Adam", "kwargs": {"lr": 3e-4}})
            InvOptim = getattr(torch.optim, inv_optim_kwargs.get("type", "Adam"))
            self.inv_optim = InvOptim(self.inverse_model.parameters(), **inv_optim_kwargs.get("kwargs", {}))

        if self.inverse_model is not None:
            self.inverse_model.eval()

    def get_discriminator_inputs(self, obs, actions, next_obs):
        if self.input_type == "state":
            input_1, input_2 = obs, None
        elif self.input_type == "state_action":
            input_1, input_2 = obs, actions
        elif self.input_type == "state_state":
            input_1, input_2 = obs, next_obs
        else:
            raise NotImplementedError

        return input_1, input_2

    def _normalize_obs_dict(self, obs):
        if self.normalize_input:
            return {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
        return obs

    def _mask_obs_dict(self, obs, mask, mask_cpu):
        masked = {}
        for k, v in obs.items():
            use_mask = mask_cpu if v.device.type == "cpu" else mask
            masked[k] = v[use_mask]
        return masked

    def _next_obs_from_indices(self, flat_indices):
        # Map flattened indices back to (step, env) for next_obs lookup.
        steps_per_env = self.storage.transitions_per_env
        flat_indices = flat_indices.long()
        env_idx = torch.div(flat_indices, steps_per_env, rounding_mode="floor")
        step_idx = flat_indices % steps_per_env
        next_step_idx = torch.clamp(step_idx + 1, max=steps_per_env - 1)

        dones = self.storage.storage_dict["dones"][step_idx, env_idx].bool()
        valid = (step_idx < (steps_per_env - 1)) & (~dones)

        next_obs = {}
        for k, v in self.storage.storage_dict["obses"].items():
            idx_env = env_idx.to(v.device)
            idx_step = next_step_idx.to(v.device)
            next_obs[k] = v[idx_step, idx_env]
        return next_obs, valid

    def _train_inverse_model(self):
        data = self.storage.data_dict
        if data is None or self.inverse_model is None:
            return None
        total = data["actions"].shape[0]
        if total < self.inv_batch_size:
            return None

        self.inverse_model.train()
        losses = []
        best_loss = None
        bad_count = 0
        normalizer = self.obs_rms if self.normalize_input else None

        for _ in range(self.inv_train_iters):
            indices = torch.randint(0, total, (self.inv_batch_size,), device=self.device)
            obs = {k: v[indices] for k, v in data["obses"].items()}
            next_obs, valid_mask = self._next_obs_from_indices(indices)
            if not valid_mask.any():
                continue
            mask_cpu = valid_mask.cpu()
            obs = self._mask_obs_dict(obs, valid_mask, mask_cpu)
            next_obs = self._mask_obs_dict(next_obs, valid_mask, mask_cpu)
            actions = data["actions"][indices][valid_mask]

            loss = self.inverse_model.train_batch(obs, actions, next_obs, self.inv_optim, normalizer=normalizer)
            losses.append(loss)

            if self.inv_patience > 0:
                loss_val = float(loss.item())
                if best_loss is None or (best_loss - loss_val) > self.inv_min_delta:
                    best_loss = loss_val
                    bad_count = 0
                else:
                    bad_count += 1
                    if bad_count >= self.inv_patience:
                        break

        self.inverse_model.eval()
        if len(losses) == 0:
            return None
        return torch.stack(losses).mean().detach()

    @torch.no_grad()
    def gail_reward(self, obs, actions, next_obs):
        input_1, input_2 = self.get_discriminator_inputs(obs, actions, next_obs)
        logits = self.discriminator(input_1, input_2)
        probs = torch.sigmoid(logits)
        reward = -torch.log(1.0 - probs + 1e-6)
        return (reward * self.reward_scale).unsqueeze(-1)

    @torch.no_grad()
    def play_steps(self):
        # identical to PPO.play_steps except rewards come from discriminator
        for n in range(self.horizon_len):
            if not self.env_autoresets:
                if any(self.dones):
                    done_indices = torch.where(self.dones)[0].tolist()
                    obs_reset = self.env.reset_idx(done_indices)
                    obs_reset = self._convert_obs(obs_reset)
                    for k, v in obs_reset.items():
                        self.obs[k][done_indices] = v

            model_out = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs)
            for k in ['actions', 'neglogp', 'values', 'mu', 'sigma']:
                self.storage.update_data(k, n, model_out[k])

            # do env step
            actions = torch.clamp(model_out['actions'], -1.0, 1.0)
            obs, r, self.dones, infos = self.env.step(actions)
            self.obs = self._convert_obs(obs)

            # GAIL reward from discriminator using pre-step obs and taken actions
            shaped_rewards = self.gail_reward(
                obs={k: v[n] for k, v in self.storage.storage_dict['obses'].items()},
                actions=model_out['actions'],
                next_obs=self.obs,
            )

            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            if self.value_bootstrap and 'time_outs' in infos:
                time_outs = infos['time_outs']
                time_outs = time_outs.reshape(-1, 1)
                shaped_rewards += self.gamma * model_out['values'] * time_outs.float()
            self.storage.update_data('rewards', n, shaped_rewards)

            # still track environment reward for metrics
            rewards = r.reshape(-1, 1)
            done_indices = torch.where(self.dones)[0].tolist()
            self.metrics.update(self.epoch, self.env, self.obs, rewards.squeeze(-1), done_indices, infos)
        self.metrics.flush_video(self.epoch)

        model_out = self.model_act(self.obs)
        last_values = model_out['values']

        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        values = self.storage.data_dict['values']
        returns = self.storage.data_dict['returns']
        if self.normalize_value:
            self.value_rms.update(values)
            values = self.value_rms.normalize(values)
            self.value_rms.update(returns)
            returns = self.value_rms.normalize(returns)
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def _sample_batch(self, obs_rollout, actions, batch_size, next_obs_rollout=None, dones=None):
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(0)
            obs_rollout = {k: v.unsqueeze(0) for k, v in obs_rollout.items()}
        num_trajs, length, _ = actions.shape

        if length < 2:
            raise ValueError("Rollout length must be >= 2 to sample next_obs")

        if dones is not None:
            # Ensure shape consistency
            if len(dones.shape) == 1:
                dones = dones.unsqueeze(0)
            assert dones.shape[:2] == (num_trajs, length), f"dones must have shape ({num_trajs}, {length}), got {dones.shape}"

            # Valid steps are those where 'done' is False *and* next step exists
            valid_mask = ~dones[:, :-1]  # (num_trajs, length-1)
            valid_indices = valid_mask.nonzero(as_tuple=False)  # (N_valid, 2)

            if len(valid_indices) == 0:
                raise ValueError("No valid (obs, next_obs) pairs found before dones.")

            # Sample from valid indices
            sample_idx = torch.randint(0, len(valid_indices), (batch_size,), device=self.device)
            trajs_idx = valid_indices[sample_idx, 0]
            steps_idx = valid_indices[sample_idx, 1]
        else:
            # Uniform sampling over all steps except last
            trajs_idx = torch.randint(0, num_trajs, (batch_size,), device=self.device)
            steps_idx = torch.randint(0, length - 1, (batch_size,), device=self.device)

        obs = {k: v[trajs_idx, steps_idx, ...].detach() for k, v in obs_rollout.items()}
        act = actions[trajs_idx, steps_idx, :].detach()
        if next_obs_rollout is None:
            next_obs = {k: v[trajs_idx, steps_idx + 1, ...].detach() for k, v in obs_rollout.items()}
        else:
            next_obs = {k: v[trajs_idx, steps_idx, ...].detach() for k, v in next_obs_rollout.items()}

        if self.normalize_input:
            # print("Normalizing discriminator inputs")
            # print("Before normalization:")
            # print(obs['obs'][:2])
            obs = {k: self.obs_rms[k].normalize(v) for k, v in obs.items()}
            next_obs = {k: self.obs_rms[k].normalize(v) for k, v in next_obs.items()}

        return obs, act, next_obs

    def train_discriminator(self):
        # use the flattened storage prepared in prepare_training
        data = self.storage.data_dict
        obs = data['obses']
        act = data['actions']

        self.discriminator.train()
        losses = {
            "total": [],
            "real": [],
            "fake": [],
        }
        for _ in range(self.disc_iters):
            # sample policy batch
            pol_obs, pol_act, pol_next_obs = self._sample_batch(obs, act, self.policy_batch_size)

            # sample expert batch
            exp_obs, exp_act, exp_next_obs = self._sample_batch(
                self.demos["obs"],
                self.demos["act"],
                self.expert_batch_size,
                dones=self.demos["done"],
                next_obs_rollout=self.demos["next_obs"],
            )

            # forward
            pol_input_1, pol_input_2 = self.get_discriminator_inputs(pol_obs, pol_act, pol_next_obs)
            pol_logits = self.discriminator(pol_input_1, pol_input_2)

            exp_input_1, exp_input_2 = self.get_discriminator_inputs(exp_obs, exp_act, exp_next_obs)
            exp_logits = self.discriminator(exp_input_1, exp_input_2)

            # print(pol_input_1["obs"].shape, pol_input_2.shape)
            # print(exp_input_1["obs"].shape, exp_input_2.shape)

            # print(pol_input_1["obs"], pol_input_2[:2])
            # print(exp_input_1["obs"], exp_input_2[:2])

            # print("exp_logits:", exp_logits[:10].detach())
            # print("pol_logits:", pol_logits[:10].detach())

            # input("continue")

            # labels with optional smoothing
            real_label = 1.0 - self.label_smooth
            fake_label = 0.0 + self.label_smooth
            loss_real = F.binary_cross_entropy_with_logits(exp_logits, torch.full_like(exp_logits, real_label))
            loss_fake = F.binary_cross_entropy_with_logits(pol_logits, torch.full_like(pol_logits, fake_label))
            loss = loss_real + loss_fake

            self.disc_optim.zero_grad()
            loss.backward()
            self.disc_optim.step()
            losses["total"].append(loss.detach())
            losses["real"].append(loss_real.detach())
            losses["fake"].append(loss_fake.detach())

        self.discriminator.eval()
        return {
            "discriminator/total": torch.stack(losses["total"]).mean().item() if len(losses["total"]) > 0 else 0.0,
            "discriminator/real": torch.stack(losses["real"]).mean().item() if len(losses["real"]) > 0 else 0.0,
            "discriminator/fake": torch.stack(losses["fake"]).mean().item() if len(losses["fake"]) > 0 else 0.0,
        }

    def train(self):
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)

        while self.agent_steps < self.max_agent_steps:
            self.epoch += 1
            if self.job_clock is not None:
                _, safe_stop = self.job_clock.step(check_safe_stop=True)
                if safe_stop:
                    print("Not enough time left for another step. Exiting cleanly.")
                    break

            print("Collecting experience...")
            self.set_eval()
            self.play_steps()
            self.agent_steps += self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size

            disc_metrics = {}
            # discriminator update step(s)
            if self.epoch % self.gail_config.get("disc_update_freq", 1) == 0:
                print("Training discriminator...")
                disc_metrics = self.train_discriminator()

            print("Training policy...")
            self.set_train()
            results = self.train_epoch()  # reuse PPO training on shaped rewards
            self.storage.data_dict = None

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                # train metrics
                metrics = {k: torch.mean(torch.stack(v)).item() for k, v in results.items()}
                metrics.update({k: torch.mean(torch.cat(results[k]), 0).cpu().numpy() for k in ['mu', 'sigma']})
                metrics.update(
                    {
                        'epoch': self.epoch,
                        'mini_epoch': self.mini_epoch,
                        'last_lr': self.last_lr,
                        'e_clip': self.e_clip,
                        'expert_return': self.demos["expert_return"],
                    }
                )
                metrics = {f'train_stats/{k}': v for k, v in metrics.items()}

                # timing
                timings = self.timer.stats(step=self.agent_steps, total_names=self.timer_total_names, reset=False)
                timing_metrics = {f'train_timings/{k}': v for k, v in timings.items()}
                metrics.update(timing_metrics)

                # episode metrics
                episode_metrics = {
                    'train_scores/episode_rewards': self.metrics.episode_trackers['rewards'].mean(),
                    'train_scores/episode_lengths': self.metrics.episode_trackers['lengths'].mean(),
                    'train_scores/num_episodes': self.metrics.num_episodes,
                    **self.metrics.result(prefix='train'),
                    **disc_metrics,
                }
                metrics.update(episode_metrics)

                self.writer.add(self.agent_steps, metrics)
                self.writer.write()

                self._checkpoint_save(metrics['train_scores/episode_rewards'])

                if self.print_every > 0 and (self.epoch + 1) % self.print_every == 0:
                    print(
                        f'Epochs: {self.epoch + 1} |',
                        f'Agent Steps: {int(self.agent_steps):,} |',
                        f'Best: {self.best_stat if self.best_stat is not None else -float("inf"):.2f} |',
                        'Stats:',
                        f'ep_rewards {episode_metrics["train_scores/episode_rewards"]:.2f},',
                        f'ep_lengths {episode_metrics["train_scores/episode_lengths"]:.2f},',
                        f'last_sps {timings["lastrate"]:.2f},',
                        f'ExploreEnv_time {timings["agent.play_steps/total"] / 60:.1f} min,',
                        f'UpdateRL_time {timings["agent.train_epoch/total"] / 60:.1f} min,',
                        f'SPS {timings["totalrate"]:.2f} |',
                        f"Expert return: {self.demos['expert_return']:.2f}",
                    )

        timings = self.timer.stats(step=self.agent_steps)
        print(timings)

        self.save(os.path.join(self.ckpt_dir, 'final.pth'))

    def train_epoch(self):
        results = collections.defaultdict(list)
        for mini_ep in range(0, self.mini_epochs):
            self.mini_epoch += 1
            ep_kls = []

            if self.inverse_model is not None and self.mini_epoch % self.inv_update_freq == 0:
                inv_loss = self._train_inverse_model()
                if inv_loss is not None:
                    results["loss/inverse_model"].append(inv_loss)

            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, returns, actions, obs_dict = self.storage[i]
                if not isinstance(obs_dict, dict):
                    obs_dict = {'obs': obs_dict}

                if self.normalize_input:
                    input_dict = {}
                    for k, v in obs_dict.items():
                        self.obs_rms[k].update(v)
                        input_dict[k] = self.obs_rms[k].normalize(v)
                else:
                    input_dict = obs_dict
                batch_dict = {
                    'prev_actions': actions,
                    **input_dict,
                }

                model_out = self.model(batch_dict)
                action_log_probs = model_out['prev_neglogp']
                values = model_out['values']
                entropy = model_out['entropy']
                mu = model_out['mu']
                sigma = model_out['sigma']

                a_loss, clip_frac = actor_loss(
                    old_action_log_probs, action_log_probs, advantage, self.e_clip, self.use_smooth_clamp
                )
                c_loss, explained_var = critic_loss(value_preds, values, self.e_clip, returns, self.clip_value_loss)
                b_loss = bounds_loss(mu, self.bounds_type)

                a_loss, c_loss, entropy, b_loss = [torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

                inv_reg_loss = None
                if self.inverse_model is not None and self.inv_reg_coef > 0.0:
                    start, end = self.storage.last_range
                    flat_indices = torch.arange(start, end, device=self.device)
                    next_obs, valid_mask = self._next_obs_from_indices(flat_indices)
                    if valid_mask.any():
                        mask_cpu = valid_mask.cpu()
                        obs_inv = self._mask_obs_dict(obs_dict, valid_mask, mask_cpu)
                        next_obs_inv = self._mask_obs_dict(next_obs, valid_mask, mask_cpu)
                        obs_inv = self._normalize_obs_dict(obs_inv)
                        next_obs_inv = self._normalize_obs_dict(next_obs_inv)
                        with torch.no_grad():
                            inv_pred = self.inverse_model(obs_inv, next_obs_inv)
                        inv_reg_loss = F.mse_loss(mu[valid_mask], inv_pred)
                        loss = loss + self.inv_reg_coef * inv_reg_loss

                if self.dapg_config is not None:
                    demo_actor_loss, demo_nll_loss = self.update_dapg()
                    loss += demo_actor_loss

                self.optim.zero_grad()
                loss.backward() if not self.multi_gpu else self.accelerator.backward(loss)

                if self.truncate_grads:
                    if not self.multi_gpu:
                        grad_norm_all = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    else:
                        assert self.accelerator.sync_gradients
                        grad_norm_all = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                if self.multi_gpu:
                    metrics = (kl_dist, loss, a_loss, c_loss, b_loss, entropy, clip_frac, explained_var, mu, sigma)
                    metrics = self.accelerator.gather_for_metrics(metrics)
                    kl_dist, loss, a_loss, c_loss, b_loss, entropy, clip_frac, explained_var, mu, sigma = metrics

                    if self.dapg_config is not None:
                        demo_actor_loss, demo_nll_loss = self.accelerator.gather_for_metrics((demo_actor_loss, demo_nll_loss))

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())
                ep_kls.append(kl_dist)

                results['loss/total'].append(loss)
                results['loss/actor'].append(a_loss)
                results['loss/critic'].append(c_loss)
                results['loss/bounds'].append(b_loss)
                results['loss/entropy'].append(entropy)
                results['clip_frac'].append(clip_frac)
                results['explained_var'].append(explained_var)
                results['mu'].append(mu.detach())
                results['sigma'].append(sigma.detach())
                if inv_reg_loss is not None:
                    results['loss/inv_reg'].append(inv_reg_loss.detach())
                if self.truncate_grads:
                    results['grad_norm/all'].append(grad_norm_all)

                if self.dapg_config is not None:
                    results['dapg/demo_nll_loss'].append(demo_nll_loss)
                    results['dapg/demo_actor_loss'].append(demo_actor_loss)
                    results['dapg/lambda'].append(torch.tensor(self.dapg_lambda))
                    self.update_dapg_lambda()

            avg_kl = torch.mean(torch.stack(ep_kls))
            results['avg_kl'].append(avg_kl)

            if self.lr_schedule == 'kl':
                self.last_lr = self.scheduler.update(self.last_lr, avg_kl.item())
            elif self.lr_schedule == 'cos':
                self.last_lr = adjust_learning_rate_cos(
                    self.init_lr, mini_ep, self.mini_epochs, self.agent_steps, self.max_agent_steps
                )

            for param_group in self.optim.param_groups:
                param_group['lr'] = self.last_lr

        if self.lr_schedule == 'linear':
            self.last_lr = self.scheduler.update(self.agent_steps)

        return results
