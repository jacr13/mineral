import torch
import torch.nn as nn


def MLP(
    units,
    plain_last: bool = False,
    norm_type="LayerNorm",
    act_type="ReLU",
):
    layers = []

    in_size = units[0]
    for i in range(1, len(units)):
        out_size = units[i]
        l = nn.Linear(in_size, out_size)
        layers.append(l)

        if plain_last and i == len(units) - 1:
            break

        if norm_type is not None:
            module = torch.nn
            Cls = getattr(module, norm_type)
            layers.append(Cls(out_size))

        if act_type is not None:
            module = torch.nn.modules.activation
            Cls = getattr(module, act_type)
            layers.append(Cls())

        in_size = out_size
    return nn.Sequential(*layers)


# https://github.com/YanjieZe/3D-Diffusion-Policy/blob/a825d2e8b0a22922d012be8ef9ce729e0237a4fe/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py
class DP3PointNet(nn.Module):
    def __init__(
        self,
        pcd_shapes,
        node_feature_dim=0,
        global_feature_dim=64,
        local_feature_dim=None,
        block_channels=[64, 128, 256],
        pool="max",
        norm_type="LayerNorm",
        act_type="ReLU",
        plain_last=False,
        remove_last_act=True,
    ):
        super().__init__()
        assert local_feature_dim is None, print("Not implemented")
        self.node_feature_dim = node_feature_dim
        self.global_feature_dim = global_feature_dim
        self.local_feature_dim = local_feature_dim
        self.pool = pool

        D = 3 + node_feature_dim
        norm_act_kwargs = dict(norm_type=norm_type, act_type=act_type)
        self.mlp = MLP([D, *block_channels], **norm_act_kwargs)
        self.final_projection = MLP([block_channels[-1], global_feature_dim], plain_last=plain_last, **norm_act_kwargs)
        if not plain_last and remove_last_act:
            self.final_projection = self.final_projection[:-1]

    def forward(self, data):
        x, pos = data
        B, N, _ = pos.shape
        if x is not None:
            x = torch.cat([pos, x], -1)
        else:
            x = pos

        x = self.mlp(x)
        if self.pool == "max":
            # x = torch.max(x, 1)[0]
            x = torch.amax(x, 1)
        elif self.pool == "avg":
            x = torch.mean(x, 1)
        else:
            raise ValueError(self.pool)
        x = self.final_projection(x)
        return x, None
