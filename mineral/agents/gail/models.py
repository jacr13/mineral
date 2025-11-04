import torch
import torch.nn as nn

from ...nets import MLP, MultiEncoder
from ..ppo.models import weight_init_


class Discriminator(nn.Module):
    def __init__(self, obs_space, act_dim, input_type='state_action', discriminator_kwargs=None):
        super().__init__()
        if discriminator_kwargs is None:
            discriminator_kwargs = {}

        encoder_kwargs = discriminator_kwargs.get('encoder_kwargs', {})
        mlp_kwargs = discriminator_kwargs.get('mlp_kwargs', None)

        if mlp_kwargs is None:
            mlp_kwargs = {"units": [256, 256], "norm_type": "LayerNorm", "act_type": "SiLU"}

        self.obs_space = obs_space
        self.input_type = input_type

        if 'obs' in obs_space:
            self.encoder = nn.Identity()
            mlp_in_dim = obs_space['obs'][0]
        else:
            self.encoder = MultiEncoder(obs_space, encoder_kwargs, weight_init_fn=weight_init_)
            mlp_in_dim = self.encoder.out_dim

        if input_type == 'state_action':
            mlp_in_dim += act_dim
        elif input_type == 'state_state':
            mlp_in_dim += mlp_in_dim
        elif input_type == 'state':
            mlp_in_dim = mlp_in_dim
        else:
            raise NotImplementedError(input_type)

        self.net = MLP(mlp_in_dim, out_dim=1, plain_last=True, **mlp_kwargs)

    def _encode(self, obs_dict):
        if 'obs' in self.obs_space:
            z = obs_dict['obs']
        else:
            encoder_out = self.encoder(obs_dict)
            z = encoder_out['z']
        return z

    def forward(self, input_1, input_2=None):
        if input_2 is None:
            assert self.input_type == 'state', 'you should provide 2 inputs if input_type is not "state"'

        if self.input_type == 'state':
            x = self._encode(input_1)
        elif self.input_type == 'state_action':
            z1 = self._encode(input_1)
            x = torch.cat([z1, input_2], dim=-1)
        elif self.input_type == 'state_state':
            z1 = self._encode(input_1)
            z2 = self._encode(input_2)
            x = torch.cat([z1, z2], dim=-1)
        else:
            raise NotImplementedError

        logits = self.net(x)
        return logits.squeeze(-1)
