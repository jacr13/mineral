import collections
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn

from ..actorcritic_base import ActorCriticBase
from . import models
from .experience import ExperienceBuffer
from .utils import AdaptiveScheduler, LinearScheduler, RewardShaper, RunningMeanStd, adjust_learning_rate_cos


class PPO(ActorCriticBase):
    def __init__(self, env, output_dir, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.ppo_config = full_cfg.agent.ppo
        self.num_actors = self.ppo_config['num_actors']
        super().__init__(env, output_dir, full_cfg, **kwargs)

        # ---- Normalizer ----
        if self.normalize_input:
            self.running_mean_std = {
                k: RunningMeanStd(v) if re.match(self.input_keys_normalize, k) else nn.Identity()
                for k, v in self.obs_space.items()
            }
            self.running_mean_std = nn.ModuleDict(self.running_mean_std).to(self.device)
            print('RunningMeanStd:', self.running_mean_std)
        # ---- Model ----
        encoder, encoder_kwargs = self.network_config.get('encoder', None), self.network_config.get('encoder_kwargs', None)
        ModelCls = getattr(models, self.network_config.get('actor_critic', 'ActorCritic'))
        model_kwargs = self.network_config.get('actor_critic_kwargs', {})
        self.model = ModelCls(self.obs_space, self.action_dim, encoder=encoder, encoder_kwargs=encoder_kwargs, **model_kwargs)
        self.model.to(self.device)
        print(self.model, '\n')
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Optim ----
        optim_kwargs = self.ppo_config.get('optim_kwargs', {})
        learning_rate = optim_kwargs.get('lr', 3e-4)
        self.init_lr = float(learning_rate)
        self.last_lr = float(learning_rate)
        OptimCls = getattr(torch.optim, self.ppo_config.optim_type)
        self.optimizer = OptimCls(self.model.parameters(), **optim_kwargs)
        print(self.optimizer, '\n')
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.use_smooth_clamp = self.ppo_config['use_smooth_clamp']
        self.clip_value_loss = self.ppo_config['clip_value_loss']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.bounds_type = self.ppo_config['bounds_type']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.max_grad_norm = self.ppo_config['max_grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_value = self.ppo_config['normalize_value']
        self.reward_shaper = RewardShaper(**self.ppo_config['reward_shaper'])
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_cfg.test
        # ---- Scheduler ----
        self.lr_schedule = self.ppo_config['lr_schedule']
        if self.lr_schedule == 'kl':
            self.kl_threshold = self.ppo_config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(self.init_lr, self.ppo_config['max_agent_steps'])
        # ---- Snapshot
        self.save_frequency = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # --- Training ---
        self.obs = None
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon_length,
            self.batch_size,
            self.minibatch_size,
            self.obs_space,
            self.action_dim,
            self.device,
            self.obs_keys_cpu,
        )
        self.best_rewards = -10000
        self.epoch_num = -1
        self.agent_steps = 0
        self.max_agent_steps = int(self.ppo_config['max_agent_steps'])
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

        if self.multi_gpu:
            self.model = self.accelerator.prepare(self.model)
            self.optimizer = self.accelerator.prepare(self.optimizer)
            # TODO: prepare scheduler

            if self.normalize_input:
                self.running_mean_std = self.accelerator.prepare(self.running_mean_std)
            if self.normalize_value:
                self.value_mean_std = self.accelerator.prepare(self.value_mean_std)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        input_dict = {k: self.running_mean_std[k](obs_dict[k]) for k in self.running_mean_std.keys()}
        model_out = self.model.act(input_dict)
        model_out['values'] = self.value_mean_std(model_out['values'], True)
        return model_out

    def train(self):
        _t = time.time()
        _last_t = time.time()
        obs = self.env.reset()
        self.obs = self._convert_obs(obs)
        self.dones = torch.zeros((self.num_actors,), dtype=torch.bool, device=self.device)
        self.agent_steps = self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            train_result = self.train_epoch()
            self.storage.data_dict = None

            if self.lr_schedule == 'linear':
                self.last_lr = self.scheduler.update(self.agent_steps)

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                total_time = time.time() - _t
                all_sps = self.agent_steps / total_time
                last_sps = (self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size) / (
                    time.time() - _last_t
                )
                _last_t = time.time()
                info_string = (
                    f'Epoch: {self.epoch_num} | '
                    f'Agent Steps: {int(self.agent_steps):,} | SPS: {all_sps:.1f} | '
                    f'Last SPS: {last_sps:.1f} | '
                    f'Collect Time: {self.data_collect_time / 60:.1f} min | '
                    f'Train RL Time: {self.rl_train_time / 60:.1f} min | '
                    f'Current Best: {self.best_rewards:.2f}'
                )
                print(info_string)

                metrics = self.summary_stats(total_time, all_sps, train_result)
                self.write_metrics(self.agent_steps, metrics)

                mean_rewards = metrics['metrics/episode_rewards']
                ckpt_name = f'ep={self.epoch_num}_step={int(self.agent_steps // 1e6):04}m_reward={mean_rewards:.2f}'
                if self.save_frequency > 0:
                    if (self.epoch_num % self.save_frequency == 0) and (mean_rewards <= self.best_rewards):
                        self.save(os.path.join(self.ckpt_dir, ckpt_name))
                    self.save(os.path.join(self.ckpt_dir, f'last'))

                if mean_rewards > self.best_rewards:
                    print(f'saving current best_rewards={mean_rewards:.2f}')
                    # remove previous best file
                    prev_best_ckpt = os.path.join(self.ckpt_dir, f'best_reward={self.best_rewards:.2f}.pth')
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                    self.best_rewards = mean_rewards
                    self.save(os.path.join(self.ckpt_dir, f'best_reward={mean_rewards:.2f}'))

        print('max steps achieved')

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t
        # update network
        _t = time.time()
        self.set_train()

        train_result = collections.defaultdict(list)
        for mini_ep in range(0, self.mini_epochs):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, returns, actions, obs_dict = self.storage[i]
                if not isinstance(obs_dict, dict):
                    obs_dict = {'obs': obs_dict}

                input_dict = {k: self.running_mean_std[k](obs_dict[k]) for k in self.running_mean_std.keys()}
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

                self.optimizer.zero_grad()
                loss.backward() if not self.multi_gpu else self.accelerator.backward(loss)

                if self.truncate_grads:
                    if self.multi_gpu:
                        assert self.accelerator.sync_gradients
                        grad_norm_all = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    else:
                        grad_norm_all = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                if self.multi_gpu:
                    metrics = (kl_dist, loss, a_loss, c_loss, b_loss, entropy, clip_frac, explained_var, mu, sigma)
                    metrics = self.accelerator.gather_for_metrics(metrics)
                    kl_dist, loss, a_loss, c_loss, b_loss, entropy, clip_frac, explained_var, mu, sigma = metrics

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())
                ep_kls.append(kl_dist)
                train_result['loss'].append(loss)
                train_result['a_loss'].append(a_loss)
                train_result['c_loss'].append(c_loss)
                train_result['b_loss'].append(b_loss)
                train_result['entropy'].append(entropy)
                train_result['clip_frac'].append(clip_frac)
                train_result['explained_var'].append(explained_var)
                train_result['mu'].append(mu.detach())
                train_result['sigma'].append(sigma.detach())
                if self.truncate_grads:
                    train_result['grad_norm/all'].append(grad_norm_all)

            avg_kl = torch.mean(torch.stack(ep_kls))
            train_result['avg_kl'].append(avg_kl)

            if self.lr_schedule == 'kl':
                self.last_lr = self.scheduler.update(self.last_lr, avg_kl.item())
            elif self.lr_schedule == 'cos':
                self.last_lr = adjust_learning_rate_cos(
                    self.init_lr, mini_ep, self.mini_epochs, self.agent_steps, self.max_agent_steps
                )

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr

        self.rl_train_time += time.time() - _t
        return train_result

    def summary_stats(self, total_time, all_sps, train_result):
        metrics = {
            'runtime/time/env': self.data_collect_time,
            'runtime/time/rl': self.rl_train_time,
            'runtime/time/total': total_time,
            'runtime/sps/env': self.agent_steps / self.data_collect_time,
            'runtime/sps/rl': self.agent_steps / self.rl_train_time,
            'runtime/sps/total': all_sps,
            'metrics/episode_rewards': self.episode_rewards.mean(),
            'metrics/episode_lengths': self.episode_lengths.mean(),
        }
        log_dict = {
            'train/loss/total': torch.mean(torch.stack(train_result['loss'])).item(),
            'train/loss/actor': torch.mean(torch.stack(train_result['a_loss'])).item(),
            'train/loss/bounds': torch.mean(torch.stack(train_result['b_loss'])).item(),
            'train/loss/critic': torch.mean(torch.stack(train_result['c_loss'])).item(),
            'train/loss/entropy': torch.mean(torch.stack(train_result['entropy'])).item(),
            'train/avg_kl': torch.mean(torch.stack(train_result['avg_kl'])).item(),
            'train/clip_frac': torch.mean(torch.stack(train_result['clip_frac'])).item(),
            'train/explained_var': torch.mean(torch.stack(train_result['explained_var'])).item(),
            'train/grad_norm/all': torch.mean(torch.stack(train_result['grad_norm/all'])).item() if self.truncate_grads else 0,
            'train/actor_dist/mu': torch.mean(torch.cat(train_result['mu']), 0).cpu().numpy(),
            'train/actor_dist/sigma': torch.mean(torch.cat(train_result['sigma']), 0).cpu().numpy(),
            'train/last_lr': self.last_lr,
            'train/e_clip': self.e_clip,
            'train/epoch': self.epoch_num,
        }
        return {**metrics, **log_dict}

    def eval(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {k: self.running_mean_std[k](obs_dict[k]) for k in self.running_mean_std.keys()}
            mu = self.model.act(input_dict, sample=False)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            info['reward'] = r

    def play_steps(self):
        for n in range(self.horizon_length):
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
            r, self.dones = torch.tensor(r, device=self.device), torch.tensor(self.dones, device=self.device)
            rewards = r.reshape(-1, 1)

            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_shaper(rewards.clone())
            if self.value_bootstrap and 'time_outs' in infos:
                time_outs = torch.tensor(infos['time_outs'], device=self.device)
                time_outs = time_outs.reshape(-1, 1)
                shaped_rewards += self.gamma * model_out['values'] * time_outs.float()
            self.storage.update_data('rewards', n, shaped_rewards)

            done_indices = torch.where(self.dones)[0].tolist()
            save_video = (self.save_video_every > 0) and (self.epoch_num % self.save_video_every < self.save_video_consecutive)
            self.update_tracker(rewards.squeeze(-1), done_indices, infos, save_video=save_video)

        if self.save_video_every > 0:
            # saved video steps depends on horizon_length in play_steps()
            if (self.epoch_num % self.save_video_every) == (self.save_video_consecutive - 1):
                self._info_video = {f'video/{k}': np.concatenate(v, 1) for k, v in self._video_buf.items()}
                self._video_buf = collections.defaultdict(list)

        model_out = self.model_act(obs)
        last_values = model_out['values']

        self.agent_steps += self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def save(self, f):
        ckpt = {
            'model': self.model.state_dict(),
        }
        if self.normalize_input:
            ckpt['running_mean_std'] = self.running_mean_std.state_dict()
        if self.normalize_value:
            ckpt['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(ckpt, f'{f}.pth')
        # TODO: accelerator.save

    def load(self, f):
        ckpt = torch.load(f)
        self.model.load_state_dict(ckpt['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(ckpt['running_mean_std'])
        if self.normalize_value:
            self.value_mean_std.load_state_dict(ckpt['value_mean_std'])
        # TODO: accelerator.load


def smooth_clamp(x, mi, mx):
    return 1 / (1 + torch.exp((-(x - mi) / (mx - mi) + 0.5) * 4)) * (mx - mi) + mi


def actor_loss(old_action_log_probs, action_log_probs, advantage, e_clip, use_smooth_clamp):
    clamp = smooth_clamp if use_smooth_clamp else torch.clamp
    ratio = torch.exp(old_action_log_probs - action_log_probs)
    surr1 = advantage * ratio
    surr2 = advantage * clamp(ratio, 1.0 - e_clip, 1.0 + e_clip)
    a_loss = torch.max(-surr1, -surr2)

    # https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/665c32170d84b4be66722eea405a1e08b6e7f761/isaacgymenvs/learning/common_agent.py#L484
    clipped = torch.abs(ratio - 1.0) > e_clip
    clip_frac = torch.mean(clipped.float()).detach()
    return a_loss, clip_frac


def critic_loss(value_preds, values, e_clip, returns, clip_value_loss):
    if clip_value_loss:
        value_pred_clipped = value_preds + (values - value_preds).clamp(-e_clip, e_clip)
        value_losses = (values - returns) ** 2
        value_losses_clipped = (value_pred_clipped - returns) ** 2
        c_loss = torch.max(value_losses, value_losses_clipped)
    else:
        c_loss = (values - returns) ** 2

    explained_var = (1 - torch.var(returns - values) / (torch.var(returns) + 1e-8)).clamp(0, 1).detach()
    return c_loss, explained_var


def bounds_loss(mu, bounds_type, soft_bound=1.1):
    if bounds_type == 'bound':
        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
        b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        return b_loss
    elif bounds_type == 'reg':
        reg_loss = (mu * mu).sum(axis=-1)
        return reg_loss
    else:
        raise NotImplementedError(bounds_type)


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()
