"""Re-evaluate finished runs from their final checkpoint with the rerun protocol (Agent.protocol_final).

usage (from the FOCUS checkout): uv run --locked python scripts/focus/reeval.py <run_id> ...

For each run: the environment and agent are built from the run's own resolved_config.yaml (RESULTS_DIR/<run_id>/
results/), final.pth is loaded, and the shared evaluation (deterministic and stochastic, first two episodes of each of
64 envs) is run with the current code. Output: focus/results/data/reeval/<run_id>.json with both scores, the code
commit, and the checkpoint's sha256 (the checkpoint itself stays with the run).
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FOCUS = os.path.abspath(os.path.join(HERE, '../../..'))
RESULTS_DIR = os.path.expanduser('~/docker/cluster-supervisor/results')
OUT = os.path.join(FOCUS, 'results/data/reeval')


def one(rid):
    res = os.path.join(RESULTS_DIR, rid, 'results')
    ckpt = os.path.join(res, 'ckpt', 'final.pth')
    cfg_path = os.path.join(res, 'resolved_config.yaml')
    if not (os.path.exists(ckpt) and os.path.exists(cfg_path)):
        return print(rid, 'missing final.pth or resolved_config.yaml')
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    for k in ('slurm_job_id', 'cluster_run_id', 'git_commit'):
        cfg.pop(k, None)
    logdir = tempfile.mkdtemp(prefix=f'reeval_{rid[:8]}_')
    cfg.logdir = logdir
    cfg.run = 'eval'
    cfg.ckpt = ckpt
    cfg.ckpt_keys = cfg.get('ckpt_keys', '')
    cfg.wandb.mode = 'disabled'
    cfg.agent.protocol_eval = True
    cfg.agent.eval_every_steps = 0
    cfg.task.env.render = False
    cfg.device_id = 0
    from mineral.scripts.run import main as mineral_run

    mineral_run(cfg)
    det = json.load(open(os.path.join(logdir, 'scores.json')))
    sto = json.load(open(os.path.join(logdir, 'scores_stochastic.json')))
    rec = dict(
        run_id=rid,
        code_commit=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        ckpt_sha256=hashlib.sha256(open(ckpt, 'rb').read()).hexdigest(),
        agent_steps=det['agent_steps'],
        deterministic=det['eval_scores/episode_rewards'],
        stochastic=sto['eval_scores/episode_rewards'],
        lengths_deterministic=det['eval_scores/episode_lengths'],
    )
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(os.path.join(OUT, f'{rid}.json'), 'w'))
    m = lambda x: sum(x) / len(x)
    print(rid, 'deterministic', round(m(rec['deterministic']), 1), 'stochastic', round(m(rec['stochastic']), 1))


if __name__ == '__main__':
    sys.path.insert(0, os.getcwd())
    for rid in sys.argv[1:]:
        if os.path.exists(os.path.join(OUT, f'{rid}.json')):
            continue
        one(rid)
