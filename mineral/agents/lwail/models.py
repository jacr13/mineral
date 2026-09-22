import torch.nn as nn


class PhiNet(nn.Module):
    """Intention-Conditioned Value Function (ICVF) state encoder.

    Ported from LWAIL's ``network.py::PhiNet``. Provides the "dynamics-aware
    latent space" that :class:`~mineral.agents.lwail.lwail.LWAIL` embeds states
    into (via ``lwail.using_icvf: true``) before computing its Wasserstein
    critic. Upstream only ships pretrained checkpoints for a handful of D4RL
    Mujoco tasks and treats ICVF pretraining itself as an external/TODO
    codebase, so here this is a fixed feature extractor: instantiate it and
    load a matching checkpoint via ``lwail.icvf_path``, it is not trained by
    this repo.

    Note: upstream's layer-activation condition (``i + 1 < len(hidden_dims) or
    activate_final``) is true for every ``i`` in its loop range, so every
    Linear layer (including the last) is followed by an activation. That is
    reproduced here unconditionally to match upstream's actual behavior.
    """

    def __init__(self, hidden_dims, activation=nn.GELU):
        super().__init__()
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
