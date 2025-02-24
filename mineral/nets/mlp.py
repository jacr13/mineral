import torch
import torch.nn as nn


def Norm(norm_type, size, **norm_kwargs):
    if norm_type is None:
        return nn.Identity()

    module = torch.nn
    Cls = getattr(module, norm_type)
    norm = Cls(size, **norm_kwargs)
    return norm


def Act(act_type, **act_kwargs):
    if act_type is None:
        return nn.Identity()

    module = torch.nn.modules.activation
    Cls = getattr(module, act_type)
    act = Cls(**act_kwargs)
    return act


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=None,
        units=[512, 256, 128],
        dropout=None,
        dropout_kwargs=dict(inplace=False),
        where_dropout="every",
        norm_type=None,
        norm_kwargs={},
        act_type="ReLU",
        act_kwargs=dict(inplace=True),
        bias=True,
        plain_last=False,
    ):
        super().__init__()
        if out_dim is not None:
            units = [*units, out_dim]
        self.in_dim = in_dim
        self.out_dim = units[-1]
        self.units = units

        in_size = in_dim
        layers = []
        for i, out_size in enumerate(units):
            lin = nn.Linear(in_size, out_size, bias=bias)
            layers.append(lin)
            if plain_last and i == len(units) - 1:
                break

            if dropout is not None:
                add_dropout = False
                if i == 0 and where_dropout in ("every", "first"):
                    add_dropout = True
                if (i != 0 and i != len(units) - 1) and where_dropout in ("every"):
                    add_dropout = True
                if i == len(units) - 1 and where_dropout in ("every", "last"):
                    add_dropout = True
                if add_dropout:
                    dp = nn.Dropout(dropout, **dropout_kwargs)
                    layers.append(dp)
            if norm_type is not None:
                norm = Norm(norm_type, out_size, **norm_kwargs)
                layers.append(norm)
            if act_type is not None:
                act = Act(act_type, **act_kwargs)
                layers.append(act)
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
