import re

import numpy as np
import torch
import torch.nn as nn


class MultiEncoder(nn.Module):
    def __init__(self, obs_space, cfg):
        super().__init__()
        cnn_keys = cfg.get('cnn_keys', '$^')
        pcd_keys = cfg.get('pcd_keys', '$^')
        mlp_keys = cfg.get('mlp_keys', '$^')

        excluded = {}
        shapes = {k: v for k, v in obs_space.items() if k not in excluded and not k.startswith('info_')}
        print('Encoder input shapes:', shapes)
        self.cnn_shapes = {k: v for k, v in shapes.items() if (len(v) == 3 and re.match(cnn_keys, k))}
        self.pcd_shapes = {k: v for k, v in shapes.items() if (len(v) == 2 and re.match(pcd_keys, k))}
        self.mlp_shapes = {k: v for k, v in shapes.items() if (len(v) in (1, 2) and re.match(mlp_keys, k))}
        self.shapes = {**self.cnn_shapes, **self.pcd_shapes, **self.mlp_shapes}
        print('Encoder CNN shapes:', self.cnn_shapes)
        print('Encoder PCD shapes:', self.pcd_shapes)
        print('Encoder MLP shapes:', self.mlp_shapes)

        self.out_dim = 0
        self.out_dim_local = None
        if self.cnn_shapes:
            cnn, cnn_kwargs = cfg.cnn, cfg.cnn_kwargs

            if cnn == 'resnet':
                raise NotImplementedError
            else:
                raise NotImplementedError(cnn)

        if self.pcd_shapes:
            pcd, pcd_kwargs = cfg.pcd, cfg.pcd_kwargs

            if pcd == 'pointnet2':
                raise NotImplementedError
            else:
                raise NotImplementedError(pcd)
        if self.mlp_shapes:
            tensor_shape = sum([np.prod(v, dtype=int) for v in self.mlp_shapes.values()]).item()
            self.out_dim += tensor_shape
        print('Encoder out_dim:', self.out_dim)

    def cnn(self, x):
        inputs = torch.cat([x[k] for k in self.cnn_shapes], -1)
        output = self._cnn(inputs)
        output = output.reshape((output.shape[0], -1))
        return output

    def pcd(self, x):
        _pos = torch.cat([x[k] for k in self.pcd_shapes], -2)
        _x = None
        _batch = torch.arange(_pos.shape[0], device=_pos.device).repeat_interleave(_pos.shape[1])
        B, D = _pos.shape[:2]
        _pos = _pos.reshape(-1, *_pos.shape[2:])
        data = dict(x=_x, pos=_pos, batch=_batch)

        inputs = self._pcd_transforms(data)
        global_z, local_z = self._pcd(inputs)
        del inputs
        local_z = local_z.reshape(B, D, -1)
        return global_z, local_z

    def forward(self, x):
        outputs = {}
        outputs_local = None
        if self.cnn_shapes:
            outputs['cnn'] = self.cnn(x)

        if self.pcd_shapes:
            global_z, local_z = self.pcd(x)
            outputs['pcd'] = global_z
            outputs_local = local_z

        if self.mlp_shapes:
            inputs = [x[k][..., None] if len(self.shapes[k]) == 0 else x[k] for k in self.mlp_shapes]
            inputs = torch.cat([x.reshape(x.shape[0], -1) for x in inputs], -1)
            outputs['mlp'] = inputs

        outputs = torch.cat(list(outputs.values()), -1)
        return outputs, outputs_local
