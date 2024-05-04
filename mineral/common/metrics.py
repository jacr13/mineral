import collections
import functools
import re

import numpy as np
import torch
import torch.nn as nn

from .tracker import Tracker


class Metrics(nn.Module):
    def __init__(
        self,
        current_scores,
        tracker_len,
        info_keys={},  # assuming keys in obs_dict
        save_video_every=0,
        save_video_consecutive=0,
        env_render=False,
    ):
        super().__init__()

        # --- Keys ---
        self.info_keys_sum = re.compile(info_keys.get('sum', '$^'))
        self.info_keys_min = re.compile(info_keys.get('min', '$^'))
        self.info_keys_max = re.compile(info_keys.get('max', '$^'))
        self.info_keys_final = re.compile(info_keys.get('final', '$^'))
        self.info_keys_video = re.compile(info_keys.get('video', '$^'))

        self.info_keys_scalar = re.compile(info_keys.get('scalar', '$^'))

        # --- Video ---
        self.save_video_every = save_video_every
        self.save_video_consecutive = save_video_consecutive
        self.env_render = env_render

        # --- Tracking ---
        assert 'rewards' in current_scores.keys() and 'lengths' in current_scores.keys()
        self.current_scores = current_scores
        self.tracker_len = tracker_len
        self.episode_trackers = collections.defaultdict(functools.partial(Tracker, tracker_len))
        self.num_episodes = 0

        self._current_info = {}
        self._episode_info = self.episode_trackers

        self._extras = {}

        self._current_video = collections.defaultdict(list)
        self._episode_video = None

    def update(self, epoch, env, obs, rewards, done_indices, extras, update_scores=True):
        if update_scores:
            self._update_scores(epoch, env, obs, rewards, done_indices, extras)
        self._update_infos(epoch, env, obs, rewards, done_indices, extras)
        self._update_video(epoch, env, obs, rewards, done_indices, extras)

    def _update_scores(self, epoch, env, obs, rewards, done_indices, extras):
        self.current_scores['rewards'] += rewards
        self.current_scores['lengths'] += 1
        if len(done_indices) > 0:
            self.num_episodes += len(done_indices)
            self.episode_trackers['rewards'].update(self.current_scores['rewards'][done_indices])
            self.episode_trackers['lengths'].update(self.current_scores['lengths'][done_indices])
            self.current_scores['rewards'][done_indices] = 0
            self.current_scores['lengths'][done_indices] = 0

    def _update_infos(self, epoch, env, obs, rewards, done_indices, extras):
        for k, v in extras.items():
            if re.match(self.info_keys_scalar, k):
                # assert isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                self._extras[k] = v.item()

        d = self._current_info
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)

            if re.match(self.info_keys_sum, k):
                _k = f'{k}_sum'
                if _k not in d:
                    d[_k] = v.clone()
                else:
                    d[_k] += v
            if re.match(self.info_keys_min, k):
                _k = f'{k}_min'
                if _k not in d:
                    d[_k] = v.clone()
                else:
                    d[_k] = torch.fmin(d[_k], v)
            if re.match(self.info_keys_max, k):
                _k = f'{k}_max'
                if _k not in d:
                    d[_k] = v.clone()
                else:
                    d[_k] = torch.fmax(d[_k], v)
            if re.match(self.info_keys_final, k):
                _k = f'{k}_final'
                if _k not in d:
                    d[_k] = torch.zeros_like(v)
                d[_k][done_indices] = v[done_indices]

        if len(done_indices) > 0:
            for k in d.keys():
                v = d[k][done_indices].cpu().numpy()
                self._episode_info[k].update(v)

                # reset
                if 'sum' in k or 'final' in k:
                    d[k][done_indices] = 0
                else:
                    d[k][done_indices] = torch.nan

    def _update_video(self, epoch, env, obs, rewards, done_indices, infos):
        save_video = (self.save_video_every > 0) and (epoch % self.save_video_every < self.save_video_consecutive)
        if save_video:
            if self.env_render:
                v = env.render(mode='rgb_array')
                v = self._reshape_video('env_render', v)
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                self._current_video['env_render'].append(v)

            for k, v in obs.items():
                if re.match(self.info_keys_video, k):
                    v = self._reshape_video(k, v)
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    self._current_video[k].append(v)

    @staticmethod
    def _reshape_video(k, v):
        if len(v.shape) == 3:  # H, W, C
            v = v[None][None]  # -> B, T, H, W, C
        elif len(v.shape) == 4:  # B, H, W, C
            v = v[:, None, ...]  # -> B, T, H, W, C
        elif len(v.shape) == 5:  # B, T, H, W, C
            pass
        else:
            raise RuntimeError(f'Unsupported {k} shape {v.shape}')
        return v

    def flush_video(self, epoch):
        if self.save_video_every > 0:
            # saved video steps depends on horizon_len in play_steps()
            if (epoch % self.save_video_every) == (self.save_video_consecutive - 1):
                self._episode_video = {k: np.concatenate(v, 1) for k, v in self._current_video.items()}
                self._current_video = collections.defaultdict(list)

    def result(self, metrics=None, prefix=''):
        if metrics is None:
            metrics = {}

        for k, v in self._episode_info.items():
            if k in self.current_scores.keys():  # skip rewards and lengths
                continue
            metrics[f'{prefix}_infos/{k}'] = v.mean()
            # metrics[f'{prefix}_infos_std/{k}'] = v.std()

        for k, v in self._extras.items():
            metrics[f'{prefix}_extras/{k}'] = v
        self._extras.clear()

        if self._episode_video is not None:
            for k, v in self._episode_video.items():
                metrics[f'{prefix}_video/{k}'] = v
            self._episode_video = None

        return metrics

    def state_dict(self):
        state = {
            # 'current_scores': self.current_scores,
            'episode_trackers': {k: v.window for k, v in self.episode_trackers.items()},
            'num_episodes': self.num_episodes,
            # '_current_info': self._current_info,
            '_episode_info': self._episode_info,
            # '_extras': self._extras,
            # '_current_video': self._current_video,
            # '_episode_video': self._episode_video,
        }
        return state

    def load_state_dict(self, state_dict):
        # self.current_scores = state_dict['current_scores']
        for k, v in self.episode_trackers.items():
            v.window = state_dict['episode_trackers'][k]
        self.num_episodes = state_dict['num_episodes']
        # self._current_info = state_dict['_current_info']
        self._episode_info = state_dict['_episode_info']
        # self._extras = state_dict['_extras']
        # self._current_video = state_dict['_current_video']
        # self._episode_video = state_dict['_episode_video']
