import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nets import MLP, Dist


def weight_init_orthogonal_(m, gain=1.4142135623730951):  # sqrt(2)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear
                nn.init.orthogonal_(p, gain=gain)  # Weight
                # NOTE: this assumes nn.Linear(bias=True)
                nn.init.constant_(m[i + 1], 0)  # Bias


# dreamerv3
def dreamerv3_weight_init_trunc_normal_(m, fan="avg", scale=1.0, variance_factor=1.0):
    if isinstance(m, nn.Linear):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
        _fan = {'avg': (fan_in + fan_out) / 2, 'in': fan_in, 'out': fan_out}[fan]
        nn.init.trunc_normal_(m.weight.data, a=-2.0, b=2.0)
        m.weight.data *= 1.1368 * math.sqrt(variance_factor / _fan)
        m.weight.data *= scale
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear
                # Weight
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(p.data)
                _fan = {'avg': (fan_in + fan_out) / 2, 'in': fan_in, 'out': fan_out}[fan]
                nn.init.trunc_normal_(p.data, a=-2.0, b=2.0)
                p.data *= 1.1368 * math.sqrt(variance_factor / _fan)
                p.data *= scale
                nn.init.constant_(m[i + 1], 0)  # Bias


# tdmpc2
def weight_init_trunc_normal_(m, std=0.02):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -std, std)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear
                nn.init.trunc_normal_(p, std=std)  # Weight
                nn.init.constant_(m[i + 1], 0)  # Bias


def weight_init_uniform_(m, fan="avg", scale=1.0, variance_factor=1.0):
    if isinstance(m, nn.Linear):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
        fan = {'avg': (fan_in + fan_out) / 2, 'in': fan_in, 'out': fan_out}[fan]
        limit = math.sqrt(variance_factor / fan)
        nn.init.uniform_(m.weight.data, -limit, limit)
        m.weight.data *= scale
        if hasattr(m.bias, 'data'):
            nn.init.constant_(m.bias.data, 0.0)


def weight_init_(module, weight_init):
    if weight_init == None:
        pass
    elif weight_init == "orthogonal":
        module.apply(weight_init_orthogonal_)
    elif weight_init == "orthogonalg1":
        module.apply(lambda m: weight_init_orthogonal_(m, gain=1.0))
    elif weight_init == "normal":
        module.apply(weight_init_trunc_normal_)
    elif weight_init == "dreamerv3_normal":
        module.apply(dreamerv3_weight_init_trunc_normal_)
    else:
        raise NotImplementedError(weight_init)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        fixed_sigma=True,
        init_sigma=-1.0,
        mlp_kwargs=dict(norm_type="LayerNorm", act_type="ELU"),
        dist_kwargs=dict(dist_type="normal"),
        weight_init="orthogonal",
        weight_init_last_layers=False,
    ):
        super().__init__()
        self.fixed_sigma = fixed_sigma
        self.init_sigma = init_sigma

        self.actor_mlp = MLP(state_dim, **mlp_kwargs)
        self.mu = nn.Linear(self.actor_mlp.out_dim, action_dim)
        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.ones(action_dim, dtype=torch.float32), requires_grad=True)
        else:
            self.sigma = nn.Linear(self.actor_mlp.out_dim, action_dim)
        self.dist = Dist(**dist_kwargs)

        self.weight_init = weight_init
        self.weight_init_last_layers = weight_init_last_layers
        self.reset_parameters()

    def reset_parameters(self):
        weight_init_(self, self.weight_init)

        if self.fixed_sigma:
            nn.init.constant_(self.sigma, self.init_sigma)
        else:
            if self.weight_init_last_layers:
                if self.weight_init == "orthogonal" or self.weight_init == "orthogonalg1":
                    nn.init.orthogonal_(self.mu.weight, gain=0.01)
                    nn.init.orthogonal_(self.sigma.weight, gain=0.01)
                    nn.init.zeros_(self.sigma.bias)
                elif self.weight_init == "normal":
                    # nn.init.trunc_normal_(self.mu.weight, std=0.01)
                    # nn.init.trunc_normal_(self.sigma.weight, std=0.01)
                    # weight_init_uniform_(self.mu, scale=0.01)
                    # weight_init_uniform_(self.sigma, scale=0.01)
                    nn.init.zeros_(self.sigma.bias)
                elif self.weight_init == "dreamerv3_normal":
                    dreamerv3_weight_init_trunc_normal_(self.mu, scale=0.01)
                    dreamerv3_weight_init_trunc_normal_(self.sigma, scale=0.01)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["z"]
        x = self.actor_mlp(x)
        mu = self.mu(x)
        if self.fixed_sigma:
            sigma = self.sigma
        else:
            sigma = self.sigma(x)
        mu, sigma, distr = self.dist(mu, sigma)
        return mu, sigma, distr


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        mlp_kwargs=dict(act_type="ELU", norm_type="LayerNorm"),
        weight_init="orthogonal",
    ):
        super().__init__()
        self.critic_mlp = MLP(state_dim, out_dim=1, plain_last=True, **mlp_kwargs)

        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        weight_init_(self, self.weight_init)
        # # NOTE: this is commented out to maintain parity with orig. implementation
        # if self.weight_init == "orthogonal":
        #     nn.init.orthogonal_(self.critic_mlp.mlp[-1].weight, gain=1.0)

    def forward(self, x, return_type=None):
        if isinstance(x, dict):
            x = x["z"]
        x = self.critic_mlp(x)
        if return_type == "all":
            return [x]
        elif return_type == "min_and_avg":
            return x, x.clone()
        else:
            return x


class EnsembleCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_critics=1,
        n_sample=None,
        with_vmap=False,
        mlp_kwargs=dict(act_type="ELU", norm_type="LayerNorm"),
        weight_init="orthogonal",
    ):
        super().__init__()
        self.n_critics = n_critics
        self.n_sample = n_sample

        self.with_vmap = with_vmap
        critics = []
        for _ in range(n_critics):
            out_dim = 1
            critic = MLP(state_dim, out_dim=out_dim, plain_last=True, **mlp_kwargs)
            critics.append(critic)
        critics = nn.ModuleList(critics)

        if with_vmap:
            from functorch import combine_state_for_ensemble

            # from torch.func import stack_module_state

            fn, params, _ = combine_state_for_ensemble(critics)
            self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness='different')
            self.params = nn.ParameterList([nn.Parameter(p) for p in params])
            self._vmap_repr = str(critics)
        else:
            self.critics = critics

        self.weight_init = weight_init
        self.reset_parameters()

    def extra_repr(self):
        if self.with_vmap:
            s = f"(critics): Vmap - {self._vmap_repr}"
            return s
        else:
            return ""

    def reset_parameters(self):
        if self.with_vmap:
            weight_init_(self, self.weight_init)
        else:
            for critic in self.critics:
                weight_init_(critic, self.weight_init)
                if self.weight_init == "normal" or self.weight_init == "dreamerv3_normal":
                    weight_init_uniform_(critic.mlp[-1].weight, scale=1.0)

    def forward(self, x, return_type="min"):
        if isinstance(x, dict):
            x = x["z"]
        if self.with_vmap:
            Vs = self.vmap([p for p in self.params], (), x)
        else:
            Vs = torch.stack([critic(x) for critic in self.critics])

        if return_type == "all":
            return Vs

        if self.n_sample is not None:
            indices = torch.randperm(self.n_critics)[: self.n_sample].tolist()
            Vs = Vs[indices]

        if return_type == "min":
            Vs = torch.min(Vs, dim=0).values
        elif return_type == "min_and_avg":
            Vs_min = torch.min(Vs, dim=0).values
            Vs_avg = torch.mean(Vs, dim=0)
            return Vs_min, Vs_avg
        return Vs
