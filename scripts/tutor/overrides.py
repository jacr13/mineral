"""Hydra overrides of a frozen MENTOR task file (tasks/mentor/icml/dflex_<task>.yaml) for local TUTOR runs.

Agent TUTOR/DFlex (MENTOR unchanged for MENTOR teacher kinds). Stage I is replaced by Levenberg-Marquardt with forward-difference Jacobians (no simulator gradient), as in the
gradient-free Stage I of the 14 Nov ablation (s1v, LM-fd): 5 iterations, Jacobian every iteration; Humanoid and SNU
use forward differences then Broyden updates (fd-Broyden, as their source groups).
usage: python scripts/tutor/overrides.py <task> <logdir> [seed] [key=val ...]   (keys under agent.mentor)
"""
import os
import sys

import yaml

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
GF_STAGE1 = {
    'default': {'invert.jacobian': 'fd', 'invert.iters': 5, 'invert.refresh': 0, 'invert.refresh_jacobian': 'fd'},
    'humanoid': {'invert.jacobian': 'broyden', 'invert.refresh_jacobian': 'fd', 'invert.refresh': 0},
    'snu_humanoid': {'invert.jacobian': 'broyden', 'invert.refresh_jacobian': 'fd', 'invert.refresh': 5},
}


def flat(d, p=''):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, p + k + '.')
        else:
            yield p + k, v


def overrides(task, logdir, seed=10, extra=()):
    body = yaml.safe_load(open(os.path.join(CODE, f'tasks/mentor/icml/dflex_{task}.yaml')))
    m = dict(flat(body['agent']['mentor']))
    m.update(GF_STAGE1.get(task, GF_STAGE1['default']))
    for kv in extra:
        k, v = kv.split('=', 1)
        m[k] = v
    fmt = lambda v: '[' + ','.join(map(str, v)) + ']' if isinstance(v, list) else str(v)
    out = [f'task=DFlex', f'task.env.env_name={task}', 'agent=TUTOR/DFlex', f'logdir={logdir}',
           'wandb.mode=disabled', 'run=train', f'seed={seed}']
    return out + [f'agent.mentor.{k}={fmt(v)}' for k, v in m.items()]


if __name__ == '__main__':
    a = sys.argv[1:]
    seed = int(a[2]) if len(a) > 2 and '=' not in a[2] else 10
    rest = a[3:] if len(a) > 2 and '=' not in a[2] else a[2:]
    print(' '.join(overrides(a[0], a[1], seed, rest)))
