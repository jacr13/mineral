import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cnn as CNN
from . import pcd as PCD
from .mlp import MLP


class MultiEncoder(nn.Module):
    def __init__(self, obs_space, cfg, weight_init_fn=None):
        super().__init__()
        cnn_keys = cfg.get('cnn_keys', '$^')
        pcd_keys = cfg.get('pcd_keys', '$^')
        mlp_keys = cfg.get('mlp_keys', '^obs$')
        concat_keys = cfg.get('concat_keys', '^cnn$|^pcd$|^mlp$')

        excluded = {}
        shapes = {k: v for k, v in obs_space.items() if k not in excluded and not k.startswith('info_')}
        print('Encoder input shapes:', shapes)
        self.cnn_shapes = {k: v for k, v in shapes.items() if (len(v) == 3 and re.match(cnn_keys, k))}
        self.pcd_shapes = {k: v for k, v in shapes.items() if (len(v) == 2 and re.match(pcd_keys, k))}
        self.mlp_shapes = {k: v for k, v in shapes.items() if (len(v) in (1, 2) and re.match(mlp_keys, k))}
        self.shapes = {**self.cnn_shapes, **self.pcd_shapes, **self.mlp_shapes}
        self.concat_keys = re.compile(concat_keys)
        print('Encoder CNN shapes:', self.cnn_shapes)
        print('Encoder PCD shapes:', self.pcd_shapes)
        print('Encoder MLP shapes:', self.mlp_shapes)

        self.out_dim = 0
        if self.cnn_shapes:
            in_channels = sum([v[-1] for v in self.cnn_shapes.values()])
            some_shape = list(self.cnn_shapes.values())[0]
            in_size = (some_shape[0], some_shape[1], in_channels)

            cnn, cnn_kwargs = cfg.cnn, cfg.cnn_kwargs
            Cls = getattr(CNN, cnn)
            self._cnn = Cls(in_size, **cnn_kwargs)
            self.out_dim += self._cnn.out_dim

        if self.pcd_shapes:
            pcd_inputs_kwargs = cfg.get('pcd_inputs_kwargs', {})
            self._pcd_inputs = PCD.PCDInputs(self.pcd_shapes, **pcd_inputs_kwargs)

            pcd, pcd_kwargs = cfg.pcd, cfg.pcd_kwargs
            Cls = getattr(PCD, pcd)
            self._pcd = Cls(self.pcd_shapes, **pcd_kwargs)
            self.out_dim += self._pcd.global_feature_dim

        if self.mlp_shapes:
            tensor_shape = sum([np.prod(v, dtype=int) for v in self.mlp_shapes.values()]).item()

            mlp_kwargs = cfg.get('mlp_kwargs', None)
            if mlp_kwargs is not None:
                self._mlp = MLP(tensor_shape, **mlp_kwargs)
                self.out_dim += self._mlp.out_dim
            else:
                self._mlp = None
                self.out_dim += tensor_shape

        print('Encoder out_dim:', self.out_dim)

        self.weight_init = cfg.get('weight_init', None)
        self.weight_init_cnn = cfg.get('weight_init_cnn', None)
        self.weight_init_pcd = cfg.get('weight_init_pcd', None)
        self.weight_init_mlp = cfg.get('weight_init_mlp', None)
        self.weight_init_fn = weight_init_fn
        self.reset_parameters()

    def reset_parameters(self):
        if self.cnn_shapes:
            if hasattr(self._cnn, "reset_parameters"):
                self._cnn.reset_parameters()
        if self.pcd_shapes:
            if hasattr(self._pcd, "reset_parameters"):
                self._pcd.reset_parameters()
        if self.mlp_shapes:
            if self._mlp is not None:
                if hasattr(self._mlp, "reset_parameters"):
                    self._mlp.reset_parameters()

        if self.weight_init_fn is not None:
            if self.weight_init is not None:
                self.weight_init_fn(self, self.weight_init)

            if self.weight_init_cnn is not None and self.cnn_shapes:
                self.weight_init_fn(self._cnn, self.weight_init_cnn)

            if self.weight_init_pcd is not None and self.pcd_shapes:
                self.weight_init_fn(self._pcd, self.weight_init_pcd)

            if self.weight_init_mlp is not None and self.mlp_shapes:
                self.weight_init_fn(self._mlp, self.weight_init_mlp)

    def cnn(self, x):
        inputs = torch.cat([x[k] for k in self.cnn_shapes], -1)
        output = self._cnn(inputs)
        output = output.reshape((output.shape[0], -1))
        return output

    def pcd(self, x):
        inputs = {k: x[k] for k in self.pcd_shapes}
        data = self._pcd_inputs(inputs)
        outputs = self._pcd(data)
        global_z, local_z = outputs
        return global_z, local_z

    def mlp(self, x):
        inputs = [x[k][..., None] if len(self.shapes[k]) == 0 else x[k] for k in self.mlp_shapes]
        inputs = torch.cat([x.reshape(x.shape[0], -1) for x in inputs], dim=-1)
        if self._mlp is not None:
            output = self._mlp(inputs)
        else:
            output = inputs
        return output

    def forward(self, x):
        outputs = {}
        if self.cnn_shapes:
            outputs['cnn'] = self.cnn(x)

        if self.pcd_shapes:
            outputs['pcd'], outputs['pcd_local'] = self.pcd(x)

        if self.mlp_shapes:
            outputs['mlp'] = self.mlp(x)

        z = torch.cat([v for k, v in outputs.items() if re.match(self.concat_keys, k)], dim=-1)
        outputs['z'] = z
        return outputs
