import numpy as np
import torch
import torch.nn as nn

from ...nets import Dist, MultiEncoder


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        fan_out = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.41421356237)  # np.sqrt(2)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(self, in_size, units=[512, 256, 128], act_type='ELU', norm_type=None):
        super().__init__()
        self.out_dim = units[-1]
        layers = []
        for out_size in units:
            layers.append(nn.Linear(in_size, out_size))
            if norm_type is not None:
                module = torch.nn
                Cls = getattr(module, norm_type)
                layers.append(Cls(out_size))
            if act_type is not None:
                module = torch.nn.modules.activation
                Cls = getattr(module, act_type)
                layers.append(Cls())
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_space,
        action_dim,
        mlp_kwargs={},
        separate_value_mlp=True,
        fixed_sigma=True,
        actor_dist_kwargs=dict(dist='normal'),
        encoder_kwargs={},
    ):
        super().__init__()
        self.obs_space = obs_space
        self.separate_value_mlp = separate_value_mlp
        self.fixed_sigma = fixed_sigma

        if 'obs' in obs_space:
            self.encoder = nn.Identity()
            mlp_in_dim = obs_space['obs'][0]
        else:
            self.encoder = MultiEncoder(obs_space, **encoder_kwargs)
            mlp_in_dim = self.encoder.out_dim

        self.actor_mlp = MLP(in_size=mlp_in_dim, **mlp_kwargs)
        if self.separate_value_mlp:
            self.value_mlp = MLP(in_size=mlp_in_dim, **mlp_kwargs)
        out_size = self.actor_mlp.out_dim
        self.value = nn.Linear(out_size, 1)
        self.mu = nn.Linear(out_size, action_dim)
        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32), requires_grad=True)
        else:
            self.sigma = nn.Linear(out_size, action_dim)
        self.dist = Dist(**actor_dist_kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.actor_mlp.apply(weight_init)
        if self.separate_value_mlp:
            self.value_mlp.apply(weight_init)
        self.value.apply(weight_init)
        self.mu.apply(weight_init)

        # value output layer with scale 1
        # policy output layer with scale 0.01
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        if self.fixed_sigma:
            nn.init.constant_(self.sigma, 0)
        else:
            nn.init.orthogonal_(self.sigma.weight, gain=0.01)
            nn.init.zeros_(self.sigma.bias)

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        mu, sigma, distr = self.dist(mu, logstd)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act(self, obs_dict, sample=True):
        mu, logstd, value = self._actor_critic(obs_dict)
        if not sample:
            return mu
        mu, sigma, distr = self.dist(mu, logstd)
        selected_action = distr.sample()
        neglogp = -distr.log_prob(selected_action).sum(1)
        result = {
            'neglogp': neglogp,
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    def _encode(self, obs_dict):
        if 'obs' in self.obs_space:
            z = obs_dict['obs']
            z_local = None
        else:
            z, z_local = self.encoder(obs_dict)
        return z, z_local

    def _actor_critic(self, obs_dict):
        z, z_local = self._encode(obs_dict)

        x = self.actor_mlp(z)
        mu = self.mu(x)

        if self.separate_value_mlp:
            x = self.value_mlp(z)
        value = self.value(x)

        if self.fixed_sigma:
            sigma = self.sigma
        else:
            sigma = self.sigma(x)
        return mu, mu * 0 + sigma, value
