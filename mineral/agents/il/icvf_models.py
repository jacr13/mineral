import torch.nn as nn


class PhiNet(nn.Module):
    """Intention-Conditioned Value Function (ICVF) state encoder.

    Ported from LWAIL's ``network.py::PhiNet``. Provides the "dynamics-aware
    latent space" that :class:`~mineral.agents.lwail.lwail.LWAIL` embeds states
    into (via ``lwail.using_icvf: true``) before computing its Wasserstein
    critic. Upstream only ships pretrained checkpoints for a handful of D4RL
    Mujoco tasks (dimension-incompatible with this repo's dflex tasks) and
    points to a separate, unfinished "ICVF-PyTorch Repository (todo)" for the
    training code, so here it's used purely as an architecture: at inference
    time it's a fixed feature extractor loaded from (or, on first use,
    pretrained and cached to) ``lwail.icvf_path`` -- see
    ``icvf.py::load_or_pretrain_icvf``. That checkpoint's ``phi_net`` is
    produced by training an :class:`~mineral.agents.lwail.icvf.MultilinearVF`,
    whose own ``phi_net`` is itself a `PhiNet`.

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
