import math

import torch
import torch.nn as nn


@torch.no_grad()
def soft_update(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters(), strict=False):
        # tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))
        tar.mul_(1.0 - tau).add_(cur * tau)


def distl_projection(next_dist, reward, done, gamma, v_min=-10, v_max=10, num_atoms=51, support=None):
    delta_z = (v_max - v_min) / (num_atoms - 1)
    batch_size = reward.shape[0]

    target_z = (reward + (1 - done) * gamma * support.to(done.device)).clamp(min=v_min, max=v_max)
    b = (target_z - v_min) / delta_z
    lower_idx = b.floor().long()
    upper_idx = b.ceil().long()

    lower_idx[torch.logical_and((upper_idx > 0), (lower_idx == upper_idx))] -= 1
    upper_idx[torch.logical_and((lower_idx < (num_atoms - 1)), (lower_idx == upper_idx))] += 1

    proj_dist = torch.zeros_like(next_dist)
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size, device=done.device)
    offset = offset.unsqueeze(1).expand(batch_size, num_atoms).long()
    proj_dist.view(-1).index_add_(0, (lower_idx + offset).view(-1), (next_dist * (upper_idx.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (upper_idx + offset).view(-1), (next_dist * (b - lower_idx.float())).view(-1))
    return proj_dist


def weight_init_(module, weight_init):
    if weight_init is None:
        pass
    elif weight_init == "orthogonal":  # drqv2
        module.apply(weight_init_orthogonal_)
    elif weight_init == "uniform":  # original DDPG paper
        module.apply(weight_init_uniform_)
    else:
        raise NotImplementedError(weight_init)


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
