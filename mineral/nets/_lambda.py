import inspect

import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

    def __repr__(self):
        try:
            source = inspect.getsource(self.fn).strip()
        except Exception:
            source = str(self.fn)
        return f"Lambda({source})"
