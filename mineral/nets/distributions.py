import math

import torch
import torch.distributions as D
import torch.nn.functional as F

# from torch.distributions.transforms import TanhTransform


class TanhTransform(D.Transform):
    domain = D.constraints.real
    codomain = D.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(D.TransformedDistribution):
    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale

        try:
            self.base_dist = D.Normal(loc, scale)
        except (AssertionError, ValueError) as e:
            print(loc)
            print(torch.where(torch.isnan(loc)))
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=validate_args)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()
