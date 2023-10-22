from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ...nets import Dist
from .utils import weight_init_orthogonal_, weight_init_uniform_


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=None,
        units=[512, 256, 128],
        act_type="ELU",
        act_kwargs=dict(inplace=True),
        norm_type=None,
        norm_kwargs={},
        plain_last=None,
    ):
        super().__init__()
        if out_dim is not None:
            units = [*units, out_dim]
        if plain_last is None:
            plain_last = True if out_dim is not None else False
        layers = []
        for i, output_size in enumerate(units):
            layers.append(nn.Linear(in_dim, output_size))
            if plain_last and i == len(units) - 1:
                break
            if norm_type is not None:
                module = torch.nn
                Cls = getattr(module, norm_type)
                norm = Cls(output_size, **norm_kwargs)
                layers.append(norm)
            if act_type is not None:
                module = torch.nn.modules.activation
                Cls = getattr(module, act_type)
                act = Cls(**act_kwargs)
                layers.append(act)
            in_dim = output_size
        self.mlp = nn.Sequential(*layers)
        self.out_dim = units[-1]

    def forward(self, x):
        return self.mlp(x)


def create_simple_mlp(in_dim, out_dim, hidden_layers, act_type="ELU", act_kwargs=dict(inplace=True)):
    layer_nums = [in_dim, *hidden_layers, out_dim]
    model = []
    for idx, (in_f, out_f) in enumerate(zip(layer_nums[:-1], layer_nums[1:])):
        model.append(nn.Linear(in_f, out_f))
        if idx < len(layer_nums) - 2:
            module = torch.nn.modules.activation
            Cls = getattr(module, act_type)
            act = Cls(**act_kwargs)
            model.append(act)
    return nn.Sequential(*model)


class MLPNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers=None):
        super().__init__()
        if isinstance(in_dim, Sequence):
            in_dim = in_dim[0]
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        self.net = create_simple_mlp(in_dim=in_dim, out_dim=out_dim, hidden_layers=hidden_layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        tanh_policy=True,
        fixed_sigma=None,
        mlp_kwargs={},
        dist_kwargs={},
        weight_init=None,
    ):
        super().__init__()
        self.tanh_policy = tanh_policy
        self.fixed_sigma = fixed_sigma

        self.actor_mlp = MLP(state_dim, None, **mlp_kwargs)
        self.mu = nn.Linear(self.actor_mlp.out_dim, action_dim)
        if self.tanh_policy:
            pass
        else:
            if self.fixed_sigma is None:
                pass
            elif self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = nn.Linear(self.actor_mlp.out_dim, action_dim)
            self.dist = Dist(**dist_kwargs)

        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_init == "orthogonal":  # drqv2
            self.actor_mlp.apply(weight_init_orthogonal_)
            self.mu.apply(weight_init_orthogonal_)
            nn.init.orthogonal_(self.mu.weight, gain=0.01)
            if self.fixed_sigma is None:
                pass
            elif self.fixed_sigma:
                nn.init.constant_(self.sigma, 0)
            else:
                nn.init.orthogonal_(self.sigma.weight, gain=0.01)
                nn.init.zeros_(self.sigma.bias)
        elif self.weight_init == "uniform":  # original DDPG paper
            self.actor_mlp.apply(weight_init_uniform_)
            nn.init.uniform_(self.mu.weight, -0.003, 0.003)
        elif self.weight_init == None:
            pass
        else:
            raise NotImplementedError(self.weight_init)

    def forward(self, x, std=None):
        x = self.actor_mlp(x)
        mu = self.mu(x)
        if self.tanh_policy:  # DDPG
            mu = mu.tanh()
            sigma, dist = None, None
        else:  # SAC
            if self.fixed_sigma is None:
                assert std is not None
                sigma = std
            elif self.fixed_sigma:
                sigma = self.sigma
            else:
                sigma = self.sigma(x)
            mu, sigma, dist = self.dist(mu, sigma)
        return mu, sigma, dist


class EnsembleQ(nn.Module):
    def __init__(self, state_dim, action_dim, n_critics=2, mlp_kwargs={}, weight_init=None):
        super().__init__()
        self.n_critics = n_critics
        critics = []
        for _ in range(n_critics):
            q = MLP(state_dim + action_dim, out_dim=1, **mlp_kwargs)
            critics.append(q)
        self.critics = nn.ModuleList(critics)

        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        for critic in self.critics:
            if self.weight_init == "orthogonal":  # drqv2
                critic.apply(weight_init_orthogonal_)
            elif self.weight_init == "uniform":  # original DDPG paper
                critic.apply(weight_init_uniform_)
                nn.init.uniform_(critic.mlp[-1].weight, -0.003, 0.003)
            elif self.weight_init == None:
                pass
            else:
                raise NotImplementedError(self.weight_init)

    def forward(self, state, action):
        input_x = torch.cat((state, action), dim=1)
        Qs = [critic(input_x) for critic in self.critics]
        return Qs

    def get_q_min(self, state, action):
        return torch.min(*self.forward(state, action))

    def get_q_values(self, state, action):
        return self.forward(state, action)


class DistributionalDoubleQ(nn.Module):
    def __init__(self, state_dim, act_dim, v_min=-10, v_max=10, num_atoms=51, device="cuda"):
        super().__init__()
        self.device = device
        self.net_q1 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)
        self.net_q2 = MLPNet(in_dim=state_dim + act_dim, out_dim=num_atoms)

        self.z_atoms = torch.linspace(v_min, v_max, num_atoms, device=device)

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        Q1, Q2 = self.get_q1_q2(state, action)
        Q1 = torch.sum(Q1 * self.z_atoms.to(self.device), dim=1)
        Q2 = torch.sum(Q2 * self.z_atoms.to(self.device), dim=1)
        return torch.min(Q1, Q2)  # min Q value

    def get_q_values(self, state: Tensor, action: Tensor) -> Sequence[Tensor]:
        input_x = torch.cat((state, action), dim=1)
        return torch.softmax(self.net_q1(input_x), dim=1), torch.softmax(self.net_q2(input_x), dim=1)  # two Q values
