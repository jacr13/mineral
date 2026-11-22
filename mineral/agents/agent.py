import copy
import json
import os
import random
import re
import time
from contextlib import contextmanager

import numpy as np
import torch
from omegaconf import OmegaConf

from ..common.metrics import Metrics
from ..common.writer import AtalayaWriter, TensorboardWriter, WandbWriter, Writer


class Agent:
    def __init__(self, full_cfg, logdir=None, accelerator=None, datasets=None, env=None, job_clock=None):
        assert getattr(self, 'network_config', False)
        assert getattr(self, 'num_actors', False)

        self.full_cfg = full_cfg
        self.logdir = logdir
        self.job_clock = job_clock

        # --- Device ---
        self.rank = -1
        self.device = full_cfg.rl_device
        self.multi_gpu = full_cfg.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv('LOCAL_RANK', '0'))
            self.rank_size = int(os.getenv('WORLD_SIZE', '1'))

            assert accelerator is not None
            self.accelerator = accelerator
            self.device = self.accelerator.device

        # --- Datasets ---
        self.datasets = datasets

        # --- Environment ---
        self.env = env
        action_space = self.env.action_space
        self.action_dim = action_space.shape[0]
        self.env_autoresets = full_cfg.task.get('env_autoresets', True)  # set to False to explicitly call env.reset

        # --- Inputs ---
        self.normalize_input = self.network_config.get('normalize_input', False)
        self.obs_rms_keys = re.compile(self.network_config.get('obs_rms_keys', ''))
        self.cpu_obs_keys = re.compile(self.network_config.get('cpu_obs_keys', '$^'))
        self.observation_space = self.env.observation_space
        try:
            obs_space = {k: v.shape for k, v in self.observation_space.spaces.items()}
        except:
            obs_space = {'obs': self.observation_space.shape}
        self.obs_space = obs_space

        # --- Metrics ---
        self.tracker_len = full_cfg.agent.get('tracker_len', 100)
        self.metrics_kwargs = full_cfg.agent.get('metrics_kwargs', {})
        self.env_render = full_cfg.env_render
        self.metrics = self._create_metrics(self.tracker_len, self.metrics_kwargs)

        # --- Logging ---
        self.ckpt_dir = os.path.join(self.logdir, 'ckpt')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.logdir, 'tb')  # only the plain tensorboard logger writes here

        resolved_config = OmegaConf.to_container(full_cfg, resolve=True)
        loggers = list(full_cfg.get('loggers', ['atalaya']))
        backends = self._atalaya_backends(loggers)
        main_process = self.rank <= 0  # one W&B run, data.csv and info.json per run
        meta = self._run_metadata()
        writers = []
        if 'wandb' in loggers and main_process:
            writers.append(
                WandbWriter(
                    self.logdir,
                    resolved_config,
                    wandb_options=OmegaConf.to_container(full_cfg.get('wandb', {}), resolve=True),
                    name=meta.get('run_name'),
                    run_id=meta.get('run_id'),
                    group=meta.get('run_group'),
                    project=meta.get('project'),
                )
            )
        if 'tensorboard' in loggers:
            os.makedirs(self.tb_dir, exist_ok=True)
            writers.append(TensorboardWriter(self.tb_dir, resolved_config))
        if ('atalaya' in loggers or backends) and main_process:
            writers.append(
                AtalayaWriter(
                    self.logdir,  # data.csv, info.json, log.txt, event file next to ckpt/ and scores.json
                    resolved_config,
                    backends=backends,
                    backend_options=self._atalaya_backend_options(full_cfg, backends, meta),
                    **{k: meta[m] for k, m in (('name', 'run_name'), ('project', 'project')) if meta.get(m)},
                )
            )
        self.tb_summary_writer = next((w.writer for w in writers if hasattr(w, 'writer')), None)
        self.writer = Writer(writers)

        self.print_every = full_cfg.agent.get('print_every', -1)
        self.ckpt_every = full_cfg.agent.get('ckpt_every', -1)
        self.eval_every = full_cfg.agent.get('eval_every', -1)
        self.best_stat = None

        # --- Rerun evaluation protocol (shared by every agent; off by default) ---
        # protocol_eval: the final score is protocol_evaluate() of the final policy (scores.json, deterministic; the
        # stochastic policy in scores_stochastic.json). eval_every_steps > 0: the same deterministic evaluation of
        # the current policy every eval_every_steps agent steps, from step 0 (eval_curve.jsonl). All evaluations run
        # on a dedicated environment built from the run's config with the task's standard settings, so training
        # state (envs, buffers, RNG streams) is untouched. budget.json records the steps reached.
        self.protocol_eval = bool(full_cfg.agent.get('protocol_eval', False))
        self.eval_every_steps = int(full_cfg.agent.get('eval_every_steps', 0) or 0)
        self._protocol_env = None
        self._protocol_next = None
        self._protocol_t0 = None
        self._protocol_eval_seconds = 0.0
        self._protocol_stopped_by_clock = False

        # --- Training ---
        self.epoch = -1
        self.mini_epoch = -1
        self.agent_steps = 0

    def get_actions(self, obs, sample: bool = True):
        raise NotImplementedError

    def explore_env(self, env, timesteps: int, random: bool = False, sample: bool = False):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def set_train(self):
        raise NotImplementedError

    def set_eval(self):
        raise NotImplementedError

    def save(self, f):
        raise NotImplementedError

    def load(self, f):
        raise NotImplementedError

    def _run_metadata(self):
        """Run name, project and group that run.py generated (run_metadata.yaml)."""
        path = os.path.join(self.logdir, 'run_metadata.yaml')
        if not os.path.exists(path):
            return {}
        import yaml

        return yaml.safe_load(open(path)) or {}

    @staticmethod
    def _atalaya_backends(loggers):
        """['atalaya.wandb', 'atalaya.pluto'] -> ['wandb', 'pluto'], rejecting unknown names."""
        backends = [name.split('.', 1)[1] for name in loggers if name.startswith('atalaya.')]
        unknown = [n for n in loggers if n not in ('atalaya', 'tensorboard', 'wandb') and not n.startswith('atalaya.')]
        unknown += [f'atalaya.{b}' for b in backends if b not in AtalayaWriter.BACKENDS]
        if unknown:
            allowed = ['atalaya', 'tensorboard', 'wandb'] + [f'atalaya.{b}' for b in AtalayaWriter.BACKENDS]
            raise ValueError(f'loggers: unknown {unknown}; use any of {allowed}')
        if 'wandb' in loggers and 'wandb' in backends:
            raise ValueError(
                "loggers: use 'wandb' or 'atalaya.wandb', not both: they would start two W&B runs. "
                "'atalaya.wandb' adds the offline fallback and records the upload command in info.json"
            )
        if 'tensorboard' in loggers and ('atalaya' in loggers or backends):
            raise ValueError(
                "loggers: 'atalaya' already writes the TensorBoard event file to tb/; "
                "with 'tensorboard' too every curve would appear twice. Keep one of them"
            )
        return backends

    @staticmethod
    def _atalaya_backend_options(full_cfg, backends, meta):
        options = {}
        extra = OmegaConf.to_container(full_cfg.get('atalaya_backends', {}), resolve=True) or {}
        for backend in backends:
            opts = dict(extra.get(backend) or {})
            if backend == 'wandb':
                # the same wandb block as the plain wandb logger; atalaya names the run itself
                opts = {**OmegaConf.to_container(full_cfg.get('wandb', {}), resolve=True), **opts}
                for key in ('name', 'id', 'project'):
                    opts.pop(key, None)
                if not opts.get('group') and meta.get('run_group'):
                    opts['group'] = meta['run_group']
            options[backend] = opts
        return options

    def _create_metrics(self, tracker_len, metrics_kwargs):
        current_rewards = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        current_lengths = torch.zeros(self.num_actors, dtype=int, device=self.device)
        current_scores = {'rewards': current_rewards, 'lengths': current_lengths}
        metrics = Metrics(current_scores, tracker_len, **metrics_kwargs, env_render=self.env_render)
        return metrics

    @contextmanager
    def _as_metrics(self, new_metrics):
        r"""Temporarily swaps Agent.metrics to new_metrics, useful for evaluation."""
        old_metrics = self.metrics
        self.metrics = new_metrics
        try:
            yield
        finally:
            self.metrics = old_metrics

    def _checkpoint_save(self, stat, stat_name='rewards', higher_better=True, sep=''):
        if self.ckpt_every > 0 and (self.epoch + 1) % self.ckpt_every == 0:
            ckpt_name = f'epochs{sep}{self.epoch + 1}_steps{sep}{int(self.agent_steps / 1000)}k_{stat_name}{sep}{stat:.2f}'
            self.save(os.path.join(self.ckpt_dir, ckpt_name + '.pth'))
            latest_ckpt_path = os.path.join(self.ckpt_dir, 'latest.pth')
            if os.path.lexists(latest_ckpt_path):
                os.unlink(latest_ckpt_path)
            os.symlink(ckpt_name + '.pth', latest_ckpt_path)

            # Keep this checkpoint and the nine most recent epochs already on disk.
            # Scanning the directory also bounds retention after resuming a run.
            periodic_pattern = re.compile(rf'epochs{re.escape(sep)}(\d+)_steps{re.escape(sep)}\d+k_.+\.pth')
            periodic_ckpts = []
            with os.scandir(self.ckpt_dir) as entries:
                for entry in entries:
                    match = periodic_pattern.fullmatch(entry.name)
                    if match and entry.name != ckpt_name + '.pth' and entry.is_file(follow_symlinks=False):
                        periodic_ckpts.append((int(match.group(1)), entry.name))
            for _, filename in sorted(periodic_ckpts)[:-9]:
                os.remove(os.path.join(self.ckpt_dir, filename))

        better = (stat > self.best_stat if higher_better else stat < self.best_stat) if self.best_stat is not None else True
        if better:
            print(f'saving current best_{stat_name}{sep}{stat:.2f}')
            if self.best_stat is not None:
                # remove previous best file
                prev_best_ckpt = os.path.join(self.ckpt_dir, f'best_{stat_name}{sep}{self.best_stat:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
            self.best_stat = stat
            self.save(os.path.join(self.ckpt_dir, f'best_{stat_name}{sep}{self.best_stat:.2f}.pth'))

    # ------------------------------------------------------------------
    # Rerun evaluation protocol
    # ------------------------------------------------------------------
    def protocol_act(self, obs, sample=False, t=None):
        """Action of the current policy for raw (converted, unnormalised) observations ``obs``.

        ``sample=False`` is the deterministic policy (mean / mode), ``sample=True`` the stochastic one. ``t`` is the
        step index within the episode per env (for time-conditioned policies). Implemented by each agent family.
        """
        raise NotImplementedError(f'{type(self).__name__} has no protocol_act')

    @contextmanager
    def _protocol_scope(self):
        """Run with the evaluation env's global DFlex state and leave training's RNG streams untouched."""
        state = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 np.random.get_state(), random.getstate())
        dfc = None
        try:
            import dflex

            dfc = dflex.config.no_grad
        except ImportError:
            pass
        try:
            if dfc is not None and self._protocol_env is not None:
                dflex.config.no_grad = bool(getattr(self._protocol_env, 'no_grad', True))
            yield
        finally:
            if dfc is not None:
                dflex.config.no_grad = dfc  # DFlex reads this global at every step of every env
            torch.set_rng_state(state[0])
            if state[1] is not None:
                torch.cuda.set_rng_state_all(state[1])
            np.random.set_state(state[2])
            random.setstate(state[3])

    def _protocol_get_env(self):
        if self._protocol_env is None:
            from .. import envs

            cfg = copy.deepcopy(self.full_cfg)
            OmegaConf.set_struct(cfg, False)
            env_cfg = cfg.task.env
            # the task's standard settings, as the expert's 128-episode pool: method-specific changes are undone
            for k in ('early_termination', '+early_termination'):
                if k in env_cfg:
                    del env_cfg[k]
            env_cfg.numEnvs = 64
            env_cfg.no_grad = True
            env_cfg.render = False
            if cfg.task.name == 'DFlex':  # a DFlex option (Rewarped envs do not take it)
                env_cfg.stochastic_init = True
            cfg.seed = int(cfg.seed) + 12345
            suite = getattr(envs, cfg.task.get('suite', 'isaacgymenvs'))
            dfc = None
            try:
                import dflex

                dfc = dflex.config.no_grad
            except ImportError:
                pass
            with self._protocol_scope():
                self._protocol_env = suite.make_envs(cfg)
            if dfc is not None:
                dflex.config.no_grad = dfc
        return self._protocol_env

    @torch.no_grad()
    def protocol_evaluate(self, sample=False, per_env=2):
        """First ``per_env`` episodes of each of the 64 evaluation envs; returns (returns, lengths)."""
        env = self._protocol_get_env()
        with self._protocol_scope():
            obs = self._convert_obs(env.reset())
            n = env.num_envs
            ret = torch.zeros(n, device=self.device)
            length = torch.zeros(n, dtype=torch.long, device=self.device)
            counts = [0] * n
            returns, lengths = [], []
            max_len = int(getattr(env, 'episode_length', getattr(env, 'max_episode_length', 1000)))
            for _ in range(per_env * max_len + 10):
                actions = self.protocol_act(obs, sample=sample, t=length)
                o, r, d, info = env.step(torch.as_tensor(actions, device=self.device).clamp(-1.0, 1.0))
                obs = self._convert_obs(o)
                ret += torch.as_tensor(r, device=self.device).float().view(-1)
                length += 1
                done = torch.as_tensor(d, device=self.device).view(-1).bool()
                idx = done.nonzero(as_tuple=False).flatten().tolist()
                for i in self._eval_keep(idx, counts, per_env):
                    returns.append(float(ret[i]))
                    lengths.append(int(length[i]))
                ret[done] = 0.0
                length[done] = 0
                if min(counts) >= per_env:
                    break
        return returns, lengths

    def _protocol_mode(self, train):
        for name in ('actor', 'actor_encoder', 'encoder', 'model', 'policy'):
            m = getattr(self, name, None)
            if isinstance(m, torch.nn.Module):
                m.train(train)

    def protocol_begin(self):
        """Call once before the training loop."""
        self._protocol_t0 = time.time()
        self._protocol_eval_seconds = 0.0
        curve = os.path.join(self.logdir, 'eval_curve.jsonl')
        if os.path.exists(curve):  # a restarted job starts its curve again
            os.remove(curve)
        if self.eval_every_steps > 0:
            self._protocol_next = 0
            self.protocol_step()

    def protocol_step(self):
        """Call after every training iteration: periodic evaluation when the next multiple of eval_every_steps is reached."""
        if self._protocol_next is None or self.agent_steps < self._protocol_next:
            return
        t0 = time.time()
        modes = {n: m.training for n in ('actor', 'actor_encoder', 'encoder', 'model', 'policy')
                 if isinstance(m := getattr(self, n, None), torch.nn.Module)}
        self._protocol_mode(False)
        rew, lengths = self.protocol_evaluate(sample=False)
        for n, was in modes.items():
            getattr(self, n).train(was)
        dt = time.time() - t0
        self._protocol_eval_seconds += dt
        rec = {
            'agent_steps': int(self.agent_steps),
            'epoch': int(self.epoch),
            'train_seconds': time.time() - self._protocol_t0 - self._protocol_eval_seconds,
            'eval_seconds': dt,
            'mean_return': float(np.mean(rew)),
            'mean_length': float(np.mean(lengths)),
            'episode_rewards': rew,
            'episode_lengths': lengths,
        }
        with open(os.path.join(self.logdir, 'eval_curve.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')
        self.writer.add(self.agent_steps, {'eval_curve/episode_rewards': rec['mean_return'],
                                           'eval_curve/episode_lengths': rec['mean_length']})
        print(f"[protocol] eval at {int(self.agent_steps):,} steps: return {rec['mean_return']:.1f} ({dt:.0f} s)")
        while self._protocol_next <= self.agent_steps:
            self._protocol_next += self.eval_every_steps

    def protocol_clock_stop(self):
        """Record that the job clock stopped training before the budget."""
        self._protocol_stopped_by_clock = True

    def protocol_end(self, max_agent_steps):
        """Call after the training loop (before saving final.pth)."""
        if self._protocol_t0 is None:
            return
        budget = {
            'agent_steps': int(self.agent_steps),
            'max_agent_steps': int(max_agent_steps),
            'reached_budget': bool(self.agent_steps >= max_agent_steps),
            'stopped_by_clock': self._protocol_stopped_by_clock,
            'epochs': int(self.epoch),
            'train_seconds': time.time() - self._protocol_t0 - self._protocol_eval_seconds,
            'periodic_eval_seconds': self._protocol_eval_seconds,
        }
        json.dump(budget, open(os.path.join(self.logdir, 'budget.json'), 'w'), indent=2)

    def protocol_final(self):
        """Final score of the loaded (final) policy: deterministic scores.json, stochastic scores_stochastic.json."""
        self._protocol_mode(False)
        for sample, fname in ((True, 'scores_stochastic.json'), (False, 'scores.json')):
            rew, lengths = self.protocol_evaluate(sample=sample)
            scores = {
                'epoch': int(self.epoch),
                'agent_steps': int(self.agent_steps),
                'eval_scores/protocol': True,
                'eval_scores/sample': sample,
                'eval_scores/num_episodes': len(rew),
                'eval_scores/episode_rewards': rew,
                'eval_scores/episode_lengths': lengths,
            }
            json.dump(scores, open(os.path.join(self.logdir, fname), 'w'), indent=4)
            print(f'[protocol] final ({"stochastic" if sample else "deterministic"}): {np.mean(rew):.1f} '
                  f'over {len(rew)} episodes')
        self.writer.add(self.agent_steps, {'eval_scores/episode_rewards': float(np.mean(rew)),
                                           'eval_scores/num_episodes': len(rew)})
        self.writer.write()

    def _convert_obs(self, obs):
        if not isinstance(obs, dict):
            obs = {'obs': obs}

        # NOTE: copying obs dict since env.step may modify it (ie. IsaacGymEnvs)
        _obs = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                _obs[k] = torch.tensor(v, device=self.device if not re.match(self.cpu_obs_keys, k) else 'cpu')
            else:
                # assert isinstance(v, torch.Tensor)
                _obs[k] = v
        return _obs

    def _real_next_obs(self, next_obs, dones, infos):
        """The true s_{t+1} of each env.

        For envs that reset this step (``dones`` as returned by the env, before any timeout masking), the observation
        before the reset (``infos['obs_before_reset']``) instead of the reset one.
        """
        before = infos.get('obs_before_reset') if isinstance(infos, dict) else None
        if before is None:
            return next_obs
        done = dones.bool()
        if not done.any():
            return next_obs
        before = self._convert_obs(before)
        out = {}
        for k, v in next_obs.items():
            if k in before:
                b = before[k].to(device=v.device, dtype=v.dtype)
                out[k] = torch.where(done.to(v.device).view(-1, *([1] * (v.dim() - 1))), b, v)
            else:
                out[k] = v
        return out

    blowup_threshold = 1e4  # |observation| above this = a simulation that blew up (DFlex)

    @torch.no_grad()
    def _reset_nonfinite(self, obs, next_obs, real_next_obs, dones):
        """Handle simulations that blew up (non-finite or |obs| > blowup_threshold).

        A DFlex simulation that blows up never terminates on its own (comparisons with NaN are false), so its
        observation would reach the policy and the normalisers: such envs are reset. The env's true next
        observation (``real_next_obs``, the pre-reset one at episode ends) is checked too. Every affected step is
        stored as (s, s, done). Returns (next_obs, real_next_obs, dones, number of envs affected).
        """

        def bad_rows(o):
            bad = None
            for v in o.values():
                f = v.reshape(v.shape[0], -1)
                b = ~torch.isfinite(f).all(-1) | (f.abs() > self.blowup_threshold).any(-1)
                bad = b if bad is None else bad | b
            return bad

        bad_next, bad_real = bad_rows(next_obs), bad_rows(real_next_obs)
        if bad_next is None or not (bad_next.any() or bad_real.any()):
            return next_obs, real_next_obs, dones, 0
        next_obs = {k: v.clone() for k, v in next_obs.items()}
        real_next_obs = {k: v.clone() for k, v in real_next_obs.items()}
        if bad_next.any() and hasattr(self.env, 'reset'):
            ids = torch.nonzero(bad_next).flatten()
            reset = self._convert_obs(self.env.reset(env_ids=ids))
            for k in next_obs:
                next_obs[k][ids] = reset[k][ids].to(next_obs[k].dtype)
        ids = torch.nonzero(bad_next | bad_real).flatten()
        for k in real_next_obs:
            real_next_obs[k][ids] = obs[k][ids].to(real_next_obs[k].dtype)
        dones = torch.as_tensor(dones).clone()
        dones[ids.to(dones.device)] = True
        print(f'Warning: {len(ids)} env(s) with non-finite or exploding observations (reset / stored as terminal)')
        return next_obs, real_next_obs, dones, len(ids)

    @staticmethod
    def _eval_keep(done_indices, counts, per_env=2):
        """Evaluation: keep each env's first ``per_env`` episodes only (as SHAC / MENTOR).

        Counting the first N finished episodes of any env instead over-represents envs that fail quickly
        (short, low-return episodes).
        """
        keep = [i for i in done_indices if counts[i] < per_env]
        for i in done_indices:
            counts[i] += 1
        return keep

    @staticmethod
    def _handle_timeout(dones, info, timeout_keys=('time_outs', 'TimeLimit.truncated', 'truncated')):
        timeout_envs = None
        for timeout_key in timeout_keys:
            if timeout_key in info:
                timeout_envs = info[timeout_key]
                break
        if timeout_envs is not None:
            dones = dones * (~timeout_envs)
        return dones
