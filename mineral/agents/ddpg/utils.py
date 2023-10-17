import math

import torch
import torch.nn as nn


@torch.no_grad()
def soft_update(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        # tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))
        tar.mul_(1.0 - tau).add_(cur * tau)


def handle_timeout(dones, info, timeout_keys=('time_outs', 'TimeLimit.truncated')):
    timeout_envs = None
    for timeout_key in timeout_keys:
        if timeout_key in info:
            timeout_envs = info[timeout_key]
            break
    if timeout_envs is not None:
        dones = dones * (~timeout_envs)
    return dones


class RewardShaper:
    def __init__(
        self,
        fn='scale',
        scale=1.0,
    ):
        self.fn = fn
        self.scale = scale

    def __call__(self, rewards):
        if self.fn == 'scale':
            rewards *= self.scale
        else:
            raise NotImplementedError(self.fn)
        return rewards


class RunningMeanStd(nn.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, shape, epsilon=1e-4, device='cuda'):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape, device=device))
        self.register_buffer('var', torch.ones(shape, device=device))
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x):
        out = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return out

    def unnormalize(self, x):
        out = x * torch.sqrt(self.var + self.epsilon) + self.mean
        return out

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def weight_init_orthogonal_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')  # 1.41421356237
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def weight_init_uniform_(m):
    if isinstance(m, nn.Linear):
        variance_initializer_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


# https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
# this comes from SoRB, implemented in tf
def variance_initializer_(tensor, scale=1.0, mode='fan_in', distribution='uniform'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        scale /= max(1.0, fan_in)
    elif mode == 'fan_out':
        scale /= max(1.0, fan_out)
    else:
        raise ValueError(mode)

    if distribution == 'uniform':
        limit = math.sqrt(3.0 * scale)
        nn.init.uniform_(tensor, -limit, limit)
    else:
        raise ValueError(distribution)
