import torch
import torch.nn as nn

from ...nets import MLP


class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, mlp_kwargs=None):
        super().__init__()
        if mlp_kwargs is None:
            mlp_kwargs = {"units": [256, 256], "norm_type": "LayerNorm", "act_type": "SiLU"}
        self.net = MLP(obs_dim + act_dim, out_dim=1, plain_last=True, **mlp_kwargs)

    def forward(self, obs, act):
        if isinstance(obs, dict):
            obs = obs["obs"]
        x = torch.cat([obs, act], dim=-1)
        logits = self.net(x)
        return logits.squeeze(-1)
