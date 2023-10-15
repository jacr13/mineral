import torch
import torch.distributions as D
import torch.nn as nn


class Dist(nn.Module):
    def __init__(
        self,
        dist='normal',
        minstd=1.0,
        maxstd=1.0,
        validate_args=None,
    ):
        super().__init__()
        self._dist = dist
        self._minstd = minstd
        self._maxstd = maxstd
        self.validate_args = validate_args

    def forward(self, mu, logstd):
        if self._dist == 'normal':
            sigma = torch.exp(logstd)
            distr = D.Normal(mu, sigma, validate_args=self.validate_args)
        elif self._dist == 'squashed_normal':
            raise NotImplementedError
        elif self._dist == 'normal_dreamerv3':
            lo, hi = self._minstd, self._maxstd
            std = (hi - lo) * torch.sigmoid(logstd + 2.0) + lo
            mu = torch.tanh(mu)
            distr = D.Normal(mu, std, validate_args=self.validate_args)
            sigma = std
        else:
            raise NotImplementedError(self._dist)
        return mu, sigma, distr

    def __repr__(self):
        return f'Dist(dist={self._dist})'
