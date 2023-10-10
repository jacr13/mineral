import torch


def create_buffer(capacity, obs_space, action_dim, device='cuda'):
    if isinstance(capacity, int):
        capacity = (capacity,)
    buf_obs = {}
    for k, v in obs_space.items():
        buf_obs[k] = torch.empty((*capacity, *v), dtype=torch.float32, device=device)
    buf_action = torch.empty((*capacity, int(action_dim)), dtype=torch.float32, device=device)
    buf_reward = torch.empty((*capacity, 1), dtype=torch.float32, device=device)
    buf_next_obs = {}
    for k, v in obs_space.items():
        buf_next_obs[k] = torch.empty((*capacity, *v), dtype=torch.float32, device=device)
    buf_done = torch.empty((*capacity, 1), dtype=torch.bool, device=device)
    return buf_obs, buf_action, buf_next_obs, buf_reward, buf_done


class ReplayBuffer:
    def __init__(self, obs_space, action_dim: int, capacity: int, device='cpu'):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.device = device
        self.next_p = 0  # next pointer
        self.if_full = False
        self.cur_capacity = 0  # current capacity
        self.capacity = int(capacity)

        ret = create_buffer(capacity, obs_space, action_dim, device=device)
        self.buf_obs, self.buf_action, self.buf_next_obs, self.buf_reward, self.buf_done = ret

    @torch.no_grad()
    def add_to_buffer(self, trajectory):
        obs, actions, rewards, next_obs, dones = trajectory
        obs = {k: v.reshape(-1, *self.obs_space[k]) for k, v in obs.items()}
        actions = actions.reshape(-1, self.action_dim)
        rewards = rewards.reshape(-1, 1)
        next_obs = {k: v.reshape(-1, *self.obs_space[k]) for k, v in next_obs.items()}
        dones = dones.reshape(-1, 1).bool()
        p = self.next_p + rewards.shape[0]

        if p > self.capacity:
            self.if_full = True

            for k in self.buf_obs.keys():
                self.buf_obs[k][self.next_p : self.capacity] = obs[k][: self.capacity - self.next_p]
            self.buf_action[self.next_p : self.capacity] = actions[: self.capacity - self.next_p]
            self.buf_reward[self.next_p : self.capacity] = rewards[: self.capacity - self.next_p]
            for k in self.buf_next_obs.keys():
                self.buf_next_obs[k][self.next_p : self.capacity] = next_obs[k][: self.capacity - self.next_p]
            self.buf_done[self.next_p : self.capacity] = dones[: self.capacity - self.next_p]

            p = p - self.capacity
            for k in self.buf_obs.keys():
                self.buf_obs[k][0:p] = obs[k][-p:]
            self.buf_action[0:p] = actions[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            for k in self.buf_next_obs.keys():
                self.buf_next_obs[k][0:p] = next_obs[k][-p:]
            self.buf_done[0:p] = dones[-p:]
        else:
            for k in self.buf_obs.keys():
                self.buf_obs[k][self.next_p : p] = obs[k]
            self.buf_action[self.next_p : p] = actions
            self.buf_reward[self.next_p : p] = rewards
            for k in self.buf_next_obs.keys():
                self.buf_next_obs[k][self.next_p : p] = next_obs[k]
            self.buf_done[self.next_p : p] = dones

        self.next_p = p  # update pointer
        self.cur_capacity = self.capacity if self.if_full else self.next_p

    @torch.no_grad()
    def sample_batch(self, batch_size, device='cuda'):
        indices = torch.randint(self.cur_capacity, size=(batch_size,), device=device)

        obs = {k: v[indices].to(device) for k, v in self.buf_obs.items()}
        next_obs = {k: v[indices].to(device) for k, v in self.buf_next_obs.items()}
        return (
            obs,
            self.buf_action[indices].to(device),
            self.buf_reward[indices].to(device),
            next_obs,
            self.buf_done[indices].to(device).float(),
        )
