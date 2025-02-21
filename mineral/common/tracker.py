from collections import deque
from collections.abc import Sequence

import numpy as np
import scipy.stats
import torch


class Tracker:
    def __init__(self, max_len):
        self.max_len = max_len
        self.reset()

    def reset(self):
        self.window = deque([0 for _ in range(self.max_len)], maxlen=self.max_len)

    def update(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            self.window.extend(value.tolist())
        elif isinstance(value, Sequence):
            self.window.extend(value)
        else:
            self.window.append(value)

    def mean(self):
        return np.mean(self.window)

    def std(self):
        return np.std(self.window)

    def max(self):
        return np.max(self.window)

    def min(self):
        return np.min(self.window)

    def median(self):
        return np.median(self.window)

    def sum(self):
        return np.sum(self.window)

    def iqm(self):
        # trim_mean = lambda arr, p: np.mean(sorted(arr)[int(len(arr)*p):-int(len(arr)*p) or None])
        # return trim_mean(self.window, 0.25)
        return scipy.stats.trim_mean(self.window, proportiontocut=0.25)

    def __repr__(self):
        return self.window.__repr__()
