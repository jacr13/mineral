"""Check that resume checkpoints continue training exactly (agent.resume_ckpt / resume_from).

usage (from the checkout): python scripts/resume_check.py OUT_ROOT TOTAL_STEPS SPLIT_STEPS TRAIN_SECONDS TASK_FILE...

RESUME_CHECK_SET (environment, space-separated Hydra overrides) is added to every run. The methods with a gradient
penalty on their discriminator (GAIfO, MAAD, GAIL, DAC, OPOLO, LWAIL) are checked with it set to 0: PyTorch's double
backward differs in the last bit with the process's history (measured 2026-11-25:
identical parameters, inputs and RNG give a different gradient after earlier backward passes in the same process), so
a resumed run of those drifts from the uninterrupted one by rounding, then by chaos; without the penalty everything
else they train is checked bitwise.

For each task file (its settings, with the step budget set to TOTAL_STEPS and a periodic evaluation every
TOTAL_STEPS / 4 steps), five runs with the same seed:
  full     uninterrupted, TOTAL_STEPS
  part     SPLIT_STEPS, writes ckpt/resume.pth at the end            -> extend: resume_from=part, TOTAL_STEPS
  timed    stopped by a TRAIN_SECONDS training-time budget (the job-clock stop path), writes ckpt/resume.pth
           -> cont: resume_from=timed, TOTAL_STEPS, no time budget
and compares extend and cont with full (all runs with torch_deterministic=true and DISABLE_ADDMM_CUDA_LT=1): every tensor of final.pth, the final evaluation (scores.json) and the
evaluation curve (eval_curve.jsonl, without its timing fields). PASS = all bitwise equal.
"""
import json
import os
import subprocess
import sys

import torch
import yaml

sys.path.insert(0, os.getcwd())
import spawner  # noqa: E402

BUDGET_KEYS = ('shac', 'sac', 'ddpg', 'ppo', 'diffmimic')


def run(cfg, d, extra=()):
    if os.path.exists(os.path.join(d, 'scores.json')):
        return 0
    os.makedirs(d, exist_ok=True)
    cfg = dict(cfg, logdir=d)
    # deterministic algorithms: GPU atomics (e.g. in a discriminator's backward) otherwise differ in the last bits
    # between any two runs, sporadically, and training amplifies that; resuming must be exact up to that
    ov = [o.strip() for o in spawner._build_overrides(cfg, 'local', indent='')] + ['torch_deterministic=true'] + os.environ.get('RESUME_CHECK_SET', '').split() + list(extra)
    with open(os.path.join(d, 'stdout.txt'), 'w') as f:
        # DISABLE_ADDMM_CUDA_LT: cuBLASLt's fused bias kernel (F.linear on CUDA) picks its algorithm differently in
        # different processes for some shapes (LWAIL's discriminator, batch 4000 x 512 inputs: the same weights and
        # inputs give different last bits in two processes, 2026-11-25); the plain cuBLAS path is reproducible
        env = dict(os.environ, DISABLE_ADDMM_CUDA_LT='1')
        return subprocess.run(['uv', 'run', '--locked', 'python', '-m', 'mineral.scripts.run'] + ov, stdout=f,
                              stderr=subprocess.STDOUT, env=env).returncode


def curve(d):
    p = os.path.join(d, 'eval_curve.jsonl')
    if not os.path.exists(p):
        return []
    recs = [json.loads(line) for line in open(p)]
    return [{k: v for k, v in r.items() if k not in ('train_seconds', 'eval_seconds')} for r in recs]


def compare(a, b):
    """List of differences between runs a and b (empty = identical)."""
    diffs = []
    fa, fb = (torch.load(os.path.join(x, 'ckpt', 'final.pth'), map_location='cpu', weights_only=False) for x in (a, b))

    def walk(x, y, path):
        if torch.is_tensor(x):
            if not (torch.is_tensor(y) and x.shape == y.shape and torch.equal(x, y)):
                diffs.append(f'final.pth {path}')
        elif isinstance(x, dict):
            for k in set(x) | set(y or {}):
                walk(x.get(k), (y or {}).get(k), f'{path}.{k}')
        elif x != y:
            diffs.append(f'final.pth {path}: {x} vs {y}')

    walk(fa, fb, '')
    sa, sb = (json.load(open(os.path.join(x, 'scores.json'))) for x in (a, b))
    if sa['eval_scores/episode_rewards'] != sb['eval_scores/episode_rewards']:
        diffs.append(f"scores.json: {sum(sa['eval_scores/episode_rewards']) / 128:.3f} vs "
                     f"{sum(sb['eval_scores/episode_rewards']) / 128:.3f}")
    ca, cb = curve(a), curve(b)
    if ca != cb:
        diffs.append(f'eval_curve: {len(ca)} vs {len(cb)} records, first difference at step '
                     f"{next((r['agent_steps'] for r, s in zip(ca, cb) if r != s), 'end')}")
    return diffs


def main():
    out_root, total, split, secs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    ok = True
    for path in sys.argv[5:]:
        stem = os.path.splitext(os.path.basename(path))[0]
        cfg = yaml.safe_load(open(path))
        for k in ('sweep', 'spawner'):
            cfg.pop(k, None)
        cfg['wandb'] = {'mode': 'disabled'}
        cfg['run'] = 'train_eval'
        ag = cfg['agent']
        key = next(k for k in BUDGET_KEYS if k in ag and 'max_agent_steps' in ag[k])
        ag['eval_every_steps'] = total // 4
        if 'save_eval_rollouts' in ag:
            ag['save_eval_rollouts'] = False

        def with_budget(n, **agent_extra):
            c = yaml.safe_load(yaml.safe_dump(cfg))
            c['agent'][key]['max_agent_steps'] = n
            c['agent'].update(agent_extra)
            return c

        base = os.path.join(out_root, stem)
        rc = {
            'full': run(with_budget(total), f'{base}/full'),
            'part': run(with_budget(split, **{'+resume_ckpt': True}), f'{base}/part'),
            'timed': run(with_budget(total, **{'+resume_ckpt': True, '+protocol_max_train_seconds': secs}),
                         f'{base}/timed'),
        }
        rc['extend'] = run(with_budget(total, **{'+resume_ckpt': True}), f'{base}/extend',
                           [f'resume_from={base}/part/ckpt/resume.pth'])
        rc['cont'] = run(with_budget(total, **{'+resume_ckpt': True}), f'{base}/cont',
                         [f'resume_from={base}/timed/ckpt/resume.pth'])
        if any(rc.values()):
            print(stem, 'FAILED to run', rc, flush=True)
            ok = False
            continue
        stop = json.load(open(f'{base}/timed/budget.json'))
        for name in ('extend', 'cont'):
            diffs = compare(f'{base}/full', f'{base}/{name}')
            ok &= not diffs
            at = split if name == 'extend' else stop['agent_steps']
            print(f'{stem} {name} (resumed at {at:,} steps): {"PASS" if not diffs else "FAIL"}', flush=True)
            for x in diffs[:12]:
                print('   ', x)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
