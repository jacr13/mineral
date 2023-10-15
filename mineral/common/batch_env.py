import numpy as np

try:
    import torch
except ImportError:
    pass


class BatchEnv:
    def __init__(self, envs, parallel, device='numpy'):
        # assert all(len(env) == 0 for env in envs)
        assert len(envs) > 0
        self._envs = envs
        self._parallel = parallel
        self._keys = list(self.observation_space.spaces.keys())
        self._device = device

    @property
    def observation_space(self):
        return self._envs[0].observation_space

    @property
    def action_space(self):
        return self._envs[0].action_space

    def __len__(self):
        return len(self._envs)

    def reset(self):
        return self.reset_idx(range(len(self._envs)))

    def reset_idx(self, env_ids):
        obs = []
        for env_id in env_ids:
            obs.append(self._envs[env_id].reset())
        if self._parallel:
            obs = [ob() for ob in obs]

        if self._device == 'numpy':
            obs = {k: np.array([ob[k] for ob in obs]) for k in obs[0]}
        else:
            obs = {k: torch.stack([ob[k] for ob in obs]) for k in obs[0]}
        return obs

    def step(self, action):
        assert action.shape[0] == len(self._envs)
        # assert all(len(v) == len(self._envs) for v in action.values()), (
        #     len(self._envs), {k: v.shape for k, v in action.items()})

        if not isinstance(action, np.ndarray) and self._device == 'numpy':
            action = action.cpu().numpy()

        transition = []
        for i, env in enumerate(self._envs):
            act = action[i]
            transition.append(env.step(act))
        if self._parallel:
            transition = [t() for t in transition]

        obs, reward, done, info = [], [], [], []
        for x in transition:
            obs.append(x[0])
            reward.append(x[1])
            done.append(x[2])
            info.append(x[3])

        if self._device == 'numpy':
            obs = {k: np.array([ob[k] for ob in obs]) for k in obs[0]}
            info = {k: np.array([i[k] for i in info]) for k in info[0]}
        else:
            obs = {k: torch.stack([ob[k] for ob in obs]) for k in obs[0]}
            info = {k: torch.stack([i[k] for i in info]) for k in info[0]}
        return obs, reward, done, info

    def render(self, **kwargs):
        imgs = [env.render(**kwargs) for env in self._envs]
        if self._parallel:
            imgs = [x() for x in imgs]
        if isinstance(imgs[0], np.ndarray):
            imgs = np.stack(imgs)
        elif isinstance(imgs[0], torch.Tensor):
            imgs = torch.stack(imgs)
        else:
            raise RuntimeError
        return imgs

    def seed(self, seeds):
        if isinstance(seeds, int):
            seeds = list(range(seeds, seeds + len(self._envs)))

        seeded = []
        for env, seed in zip(self._envs, seeds):
            seeded.append(env.seed(seed))
        if self._parallel:
            seeded = [s() for s in seeded]
        return seeded

    def close(self):
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass

    def __bool__(self):
        return True  # Env is always truthy, despite length zero.

    def __repr__(self):
        obs_space = {k: (v.low.min(), v.high.max(), v.shape, v.dtype) for k, v in self.observation_space.spaces.items()}
        a = self.action_space
        act_space = (a.low.min(), a.high.max(), a.shape, a.dtype)

        return (
            f'{self.__class__.__name__}(\n'
            f'  num_envs={len(self)},\n'
            f'  device={self._device},\n'
            f'  observation_space={obs_space},\n'
            f'  action_space={act_space}\n)'
        )
