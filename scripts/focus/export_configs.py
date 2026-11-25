"""Resolved configuration of every method of the main comparison, per task (writer request R10, 2026-11-25).

usage: python scripts/focus/export_configs.py CFG_ROOT [--batch p3main]

CFG_ROOT is the mineral/cfgs folder of a checkout at the batch's code commit (p3main: f4570f7), e.g. a detached
`git worktree`. For each (task, arm) of the batch's FOCUS main arms and baselines, the overrides of the seed-100 job
(the supervisor's snapshot data/runs/<run_id>/job.sbatch, i.e. exactly what ran) are composed with Hydra against
CFG_ROOT, as mineral.scripts.run does, and written to results/configs/<batch>/<task>/<arm>.yaml with the run ID.
Per-run fields that do not describe the method (seed, logdir, wandb, max_runtime) are dropped.
"""
import argparse
import csv
import os
import re
import shlex
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
FOCUS = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SUP = os.path.expanduser('~/docker/cluster-supervisor/data/runs')
DROP = ('seed=', 'logdir=', 'wandb.', 'max_runtime=')


def overrides(job):
    text = open(job).read()
    m = re.search(r"bash -c '(.*)'\s*$", text, re.S)
    inner = m.group(1).replace("'\"'\"'", "'")
    toks = shlex.split(inner.replace('\\\n', ' '))
    i = toks.index('mineral.scripts.run')
    return [t for t in toks[i + 1:] if not t.startswith(DROP)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfg_root')
    ap.add_argument('--batch', default='p3main')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(FOCUS, 'results/data/manifest.tsv')), delimiter='\t'))
    cells = {}
    for r in rows:
        if r['batch'] != a.batch or r['seed'] != '100' or '__n' in r['arm']:
            continue
        if r['arm'] in ('f_main_ot', 'f_main_dtw', 'f_main_noalign') or r['arm'].startswith('bl_'):
            cells[(r['task'], r['arm'])] = r['run_id']  # latest row of the cell = the run that counts
    n = 0
    with initialize_config_dir(config_dir=os.path.abspath(a.cfg_root), version_base='1.3'):
        for (task, arm), rid in sorted(cells.items()):
            job = os.path.join(SUP, rid, 'job.sbatch')
            if not os.path.exists(job):
                print(f'missing job script: {task} {arm} {rid}', file=sys.stderr)
                continue
            cfg = compose(config_name='config', overrides=overrides(job))
            out = os.path.join(FOCUS, 'results/configs', a.batch, task)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, f'{arm}.yaml'), 'w') as f:
                f.write(f'# {a.batch} {task} {arm}: resolved from the overrides of run {rid} (seed 100)\n')
                f.write(OmegaConf.to_yaml(cfg, resolve=False))
            n += 1
    print(f'{n} configurations in results/configs/{a.batch}')


if __name__ == '__main__':
    main()
