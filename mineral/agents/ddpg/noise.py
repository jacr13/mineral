from typing import List, Optional

import torch
from torch.distributions import Normal


class FixedNormalActionNoise:
    def __init__(self, mean, std, bounds=None):
        self._mu = mean
        self._std = std
        self._bounds = bounds
        self.dist = Normal(self._mu, self._std)

    def __call__(self, num=torch.Size(), truncated=False):
        sample = self.dist.rsample((num,))
        if truncated:
            sample.clamp(self._bounds[0], self._bounds[1])
        return sample


# @torch.jit.script
def add_normal_noise(
    x,
    std: float,
    noise_bounds: Optional[List[float]] = None,
    out_bounds: Optional[List[float]] = None,
):
    noise = torch.normal(
        torch.zeros(x.shape, dtype=x.dtype, device=x.device),
        torch.full(x.shape, std, dtype=x.dtype, device=x.device),
    )
    if noise_bounds is not None:
        noise = noise.clamp(noise_bounds[0], noise_bounds[1])
    out = x + noise
    if out_bounds is not None:
        out = out.clamp(out_bounds[0], out_bounds[1])
    return out


# @torch.jit.script
def add_mixed_normal_noise(
    x,
    std_max: float,
    std_min: float,
    noise_bounds: Optional[List[float]] = None,
    out_bounds: Optional[List[float]] = None,
):
    std_seq = torch.linspace(std_min, std_max, x.shape[0]).to(x.device).unsqueeze(-1).expand(x.shape)
    noise = torch.normal(torch.zeros(x.shape, dtype=x.dtype, device=x.device), std_seq)
    if noise_bounds is not None:
        noise = noise.clamp(noise_bounds[0], noise_bounds[1])
    out = x + noise
    if out_bounds is not None:
        out = out.clamp(out_bounds[0], out_bounds[1])
    return out
