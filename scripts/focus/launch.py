"""Launch one batch of the design and write its manifest rows (the run list is never edited by hand).

usage (from the FOCUS checkout):
  python scripts/focus/launch.py <batch> [--dry-run] [--priority 50] [--runtime 12h] [--gpu_type TYPE]
  python scripts/focus/launch.py <batch> --register-from-index <run_index.tsv>   (batch launched by enqueue.sh)

1. Generates tasks/focus/<batch>/<task>_<arm>.yaml from make_task_files.py and refuses to continue if they differ
   from the committed files (the task files of a launched batch are in git).
2. Refuses a dirty tree for code (mineral/), so the commit stored with each run is the code that ran.
3. Enqueues one run per (task, arm, seed) through spawner.py --enqueue and appends a row to
   focus/results/data/manifest.tsv: batch task arm seed run_id commit task_file budget enqueued_utc.
   A (batch, task, arm, seed) already in the manifest is never enqueued again (a seed counts once); relaunching one
   needs --relaunch "<reason>", which is recorded in the row.
"""

import argparse
import datetime
import os
import re
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_task_files as mtf  # noqa: E402

FOCUS = os.path.abspath(os.path.join(HERE, '../../..'))  # .../focus
MANIFEST = os.path.join(FOCUS, 'results/data/manifest.tsv')
RUN_INDEX = os.path.join(FOCUS, 'agent/run_index.tsv')
COLS = ['batch', 'task', 'arm', 'seed', 'run_id', 'commit', 'task_file', 'budget', 'enqueued_utc', 'note']
IMAGE = '/home/users/c/candidor/docker/mineral_uv.sif'
SUPERVISOR = '/home/joao/docker/cluster-supervisor/data'


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, check=True).stdout.strip()


def read_manifest():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST) as f:
        lines = f.read().splitlines()
    return [dict(zip(COLS, line.split('\t'))) for line in lines[1:] if line]


def append_manifest(row):
    new = not os.path.exists(MANIFEST)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, 'a') as f:
        if new:
            f.write('\t'.join(COLS) + '\n')
        f.write('\t'.join(str(row.get(c, '')) for c in COLS) + '\n')


def cells(batch):
    design = yaml.safe_load(open(os.path.join(HERE, 'design.yaml')))[batch]
    seeds = yaml.safe_load(open(os.path.join(HERE, 'seeds.yaml')))[design['seeds']]
    seeds = seeds[: int(design.get('n_seeds', len(seeds)))]
    skip = design.get('skip', {}) or {}
    for task in design['tasks']:
        for arm in design['arms']:
            if arm in skip.get(task, []):
                continue
            for seed in seeds:
                yield task, arm, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--priority', type=int, default=50)
    ap.add_argument('--runtime', default='12h')
    ap.add_argument('--gpu_type', default=None)
    ap.add_argument('--memory', default=None, help='host memory in GB per job (spawner default 8)')
    ap.add_argument('--memory_arm', action='append', default=[], metavar='ARM=GB',
                    help='host memory for one arm, e.g. bl_oops_it1000=32 (OOPS was OOM-killed at 8 GB)')
    ap.add_argument('--relaunch', default=None, help='reason; allows enqueuing cells already in the manifest')
    ap.add_argument('--register-from-index', default=None)
    ap.add_argument('--only', nargs='*', default=None, metavar='TASK:ARM:SEED', help='enqueue only these cells')
    ap.add_argument('--commit', default=None, help='code commit (default HEAD); relaunches keep the batch commit')
    args = ap.parse_args()

    todo = list(cells(args.batch))
    if args.only:
        want = {tuple(x.split(':')) for x in args.only}
        todo = [c for c in todo if (c[0], c[1], str(c[2])) in want]
    done = {(r['batch'], r['task'], r['arm'], r['seed']) for r in read_manifest()}

    if args.register_from_index:
        # rows of a batch launched with agent/enqueue.sh: "focus <batch> <task> <arm> seed=N run_id commit"
        want = {(t, a, str(s)) for t, a, s in todo}
        n = 0
        for line in open(args.register_from_index):
            m = re.match(rf'^focus {re.escape(args.batch)} (\S+) (\S+) seed=(\d+) (\S+) (\S+)$', line.strip())
            if not m or (m[1], m[2], m[3]) not in want or (args.batch, m[1], m[2], m[3]) in done:
                continue
            append_manifest(dict(batch=args.batch, task=m[1], arm=m[2], seed=m[3], run_id=m[4], commit=m[5],
                                 task_file=f'tasks/focus/{args.batch}/{m[1]}_{m[2]}.yaml',
                                 budget=mtf.TASKS[m[1]][1], enqueued_utc='', note='registered from run_index'))
            n += 1
        print(f'registered {n} runs of {args.batch}')
        return

    os.chdir(os.path.join(FOCUS, 'code'))
    if git('status', '--porcelain', 'mineral') and not args.dry_run and not args.commit:
        sys.exit('uncommitted changes in mineral/: commit them first (the stored commit must be the code that ran)')
    # generate task files and check they are committed
    arms = sorted({a for _, a, _ in todo})
    subprocess.run([sys.executable, os.path.join(HERE, 'make_task_files.py'), args.batch, *arms,
                    '--tasks', *sorted({t for t, _, _ in todo})], check=True)
    if git('status', '--porcelain', f'tasks/focus/{args.batch}') and not args.dry_run:
        sys.exit(f'generated task files of {args.batch} differ from git: commit tasks/focus/{args.batch} first')
    commit = git('rev-parse', args.commit or 'HEAD')

    n = 0
    for task, arm, seed in todo:
        key = (args.batch, task, arm, str(seed))
        if key in done and not args.relaunch:
            continue
        env_file = f'{args.batch}/{task}_{arm}.yaml'
        cmd = ['uv', 'run', '--locked', 'python', 'spawner.py', '--task_name', 'focus', '--env_files', env_file,
               '--deployment', 'slurm', '--docker', '--docker_image', IMAGE,
               '--gpu_constraint', 'COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE', '--supervisor_dir', SUPERVISOR,
               '--no-deploy_now', '--priority', str(args.priority), '--runtime', args.runtime,
               '--git_commit', commit, '--git_branch', 'focus-wip', '--set', f'seed={seed}']
        if args.gpu_type:
            cmd += ['--gpu_type', args.gpu_type]
        mem = dict(x.split('=') for x in args.memory_arm).get(arm, args.memory)
        if mem:
            cmd += ['--memory', str(mem)]
        if args.dry_run:
            print('DRY', ' '.join(cmd))
            n += 1
            continue
        out = subprocess.run(cmd + ['--enqueue'], capture_output=True, text=True)
        m = re.search(r'^Enqueued run (\S+?):', out.stdout + out.stderr, re.M)
        if not m:
            sys.exit(f'enqueue failed for {key}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}')
        row = dict(batch=args.batch, task=task, arm=arm, seed=seed, run_id=m[1], commit=commit[:7], task_file=env_file,
                   budget=mtf.TASKS[task][1], enqueued_utc=datetime.datetime.utcnow().isoformat(timespec='seconds'),
                   note=f'relaunch: {args.relaunch}' if key in done else '')
        append_manifest(row)
        with open(RUN_INDEX, 'a') as f:
            f.write(f'focus {args.batch} {task} {arm} seed={seed} {m[1]} {commit[:7]}\n')
        print(args.batch, task, arm, seed, m[1])
        n += 1
    print(f'{n} runs {"listed" if args.dry_run else "enqueued"} for {args.batch}')


if __name__ == '__main__':
    main()
