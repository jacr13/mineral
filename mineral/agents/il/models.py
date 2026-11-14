import torch
import torch.nn as nn

from ...nets import MLP, MultiEncoder
from ..ppo.models import weight_init_


class ObsEncoder(nn.Module):
    """Observation dict -> feature vector: the 'obs' tensor itself, or a MultiEncoder's 'z' for dict observations."""

    def __init__(self, obs_space, encoder_kwargs=None):
        super().__init__()
        self.flat = 'obs' in obs_space
        if self.flat:
            self.encoder = nn.Identity()
            self.out_dim = obs_space['obs'][0]
        else:
            self.encoder = MultiEncoder(obs_space, encoder_kwargs or {}, weight_init_fn=weight_init_)
            self.out_dim = self.encoder.out_dim

    def forward(self, obs):
        if self.flat:
            return obs['obs']
        return self.encoder(obs)['z']


class TransitionDiscriminator(nn.Module):
    """D(s, s') (state-only, ``use_actions=False``) or D(s, a) (``use_actions=True``); returns logits.

    With ``absorbing=True`` (DAC) every state carries an absorbing flag: an absorbing state's features are zeroed and
    the flag appended, so (s, absorbing) and (absorbing, absorbing) pairs can be scored.
    """

    def __init__(self, obs_space, action_dim=0, use_actions=False, absorbing=False, mlp_kwargs=None, encoder_kwargs=None):
        super().__init__()
        if mlp_kwargs is None:
            mlp_kwargs = {"units": [256, 256], "norm_type": "LayerNorm", "act_type": "SiLU"}
        self.use_actions = use_actions
        self.absorbing = absorbing
        self.encoder = ObsEncoder(obs_space, encoder_kwargs)
        d = self.encoder.out_dim
        in_dim = d + (action_dim if use_actions else d) + (2 if absorbing else 0)
        self.net = MLP(in_dim, out_dim=1, plain_last=True, **mlp_kwargs)

    def features(self, obs, second, abs_1=None, abs_2=None):
        """MLP input for a batch of pairs; ``second`` is next_obs (dict) or actions (tensor)."""
        z1 = self.encoder(obs)
        z2 = second if self.use_actions else self.encoder(second)
        if not self.absorbing:
            return torch.cat([z1, z2], -1)
        zeros = torch.zeros(z1.shape[0], device=z1.device)
        a1 = zeros if abs_1 is None else abs_1.float()
        a2 = zeros if abs_2 is None else abs_2.float()
        z1 = z1 * (1 - a1)[:, None]
        z2 = z2 * (1 - a2)[:, None]
        return torch.cat([z1, a1[:, None], z2, a2[:, None]], -1)

    def logits_from_features(self, x):
        return self.net(x).squeeze(-1)

    def forward(self, obs, second, abs_1=None, abs_2=None):
        return self.logits_from_features(self.features(obs, second, abs_1, abs_2))


def gradient_penalty(disc, x_expert, x_policy, target=1.0):
    """Gradient penalty (WGAN-GP, as in DAC) on interpolations of the discriminator's MLP inputs."""
    n = min(x_expert.shape[0], x_policy.shape[0])
    x_e, x_p = x_expert[:n].detach(), x_policy[:n].detach()
    eps = torch.rand(n, 1, device=x_e.device)
    x = (eps * x_e + (1 - eps) * x_p).requires_grad_(True)
    out = disc.logits_from_features(x)
    (grad,) = torch.autograd.grad(out.sum(), x, create_graph=True)
    return ((grad.norm(2, dim=-1) - target) ** 2).mean()


class Discriminator(nn.Module):
    """Single-input critic f(s) / f(s, a) / f(s, s') (LWAIL's Wasserstein critic)."""

    def __init__(self, obs_space, act_dim, input_type, discriminator_kwargs=None, encoder=None, encoder_kwargs=None):
        super().__init__()
        discriminator_kwargs = discriminator_kwargs or {}
        mlp_kwargs = discriminator_kwargs.get('mlp_kwargs', None)
        if mlp_kwargs is None:
            mlp_kwargs = {"units": [256, 256], "norm_type": "LayerNorm", "act_type": "SiLU"}
        self.input_type = input_type
        self.encoder = ObsEncoder(obs_space, encoder_kwargs)
        d = self.encoder.out_dim
        if input_type == 'state_action':
            in_dim = d + act_dim
        elif input_type == 'state_state':
            in_dim = 2 * d
        elif input_type == 'state':
            in_dim = d
        else:
            raise NotImplementedError(input_type)
        self.net = MLP(in_dim, out_dim=1, plain_last=True, **mlp_kwargs)

    def forward(self, input_1, input_2=None):
        z1 = self.encoder(input_1)
        if self.input_type == 'state':
            x = z1
        elif self.input_type == 'state_action':
            x = torch.cat([z1, input_2], dim=-1)
        else:
            x = torch.cat([z1, self.encoder(input_2)], dim=-1)
        return self.net(x).squeeze(-1)
