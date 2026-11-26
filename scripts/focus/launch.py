"""Launch one batch of the design and write its manifest rows (the run list is never edited by hand).

usage (from the FOCUS or MENTOR checkout; the project is the checkout's folder, see project.py):
  python scripts/focus/launch.py <batch> [--dry-run] [--priority 50] [--runtime 12h] [--gpu_type TYPE]
  python scripts/focus/launch.py <batch> --register-from-index <run_index.tsv>   (batch launched by enqueue.sh)

1. Generates <task_dir>/<batch>/<task>_<arm>.yaml from the project's make_task_files.py and refuses to continue if they differ
   from the committed files (the task files of a launched batch are in git).
2. Refuses a dirty tree for code (mineral/), so the commit stored with each run is the code that ran.
3. Enqueues one run per (task, arm, seed) through spawner.py --enqueue and appends a row to
   <project>/results/data/manifest.tsv: batch task arm seed run_id commit task_file budget enqueued_utc.
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
import project  # noqa: E402

mtf = project.task_generator()
P = project.P
MANIFEST = project.MANIFEST
RUN_INDEX = P['run_index']
TASK_DIR, TAG = P['task_dir'], P['tag']
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
    design = yaml.safe_load(open(project.DESIGN))[batch]
    seeds = yaml.safe_load(open(project.SEEDS))[design['seeds']]
    seeds = seeds[: int(design.get('n_seeds', len(seeds)))]
    skip = design.get('skip', {}) or {}
    for task in design['tasks']:
        for arm in design['arms']:
            if arm in skip.get(task, []):
                continue
            for seed in seeds:
                yield task, arm, seed


# Supervisor pilots (measured 25-26 Nov: FOCUS Hopper 4 per V100 at 2.4x throughput, Humanoid gains nothing):
# task -> (gpu_share, expected hours when sharing a GPU, GPU memory GB). Humanoid / SNU Humanoid fill a GPU alone.
PACK = {'cartpoleswingup': (0.3, 1.0, 1.5), 'hopper': (0.3, 2.5, 1.5), 'cheetah': (0.5, 2.5, 1.5), 'ant': (0.5, 3.0, 1.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--priority', type=int, default=50)
    ap.add_argument('--runtime', default='12h')
    ap.add_argument('--gpu_type', default=None)
    ap.add_argument('--cluster', default=None, help='supervisor cluster (bamboo, yggdrasil); default baobab')
    ap.add_argument('--gpu_constraint', default='COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE',
                    help='Slurm --constraint; Bamboo: COMPUTE_TYPE_AMPERE (RTX 3090, A100), Yggdrasil: '
                    'COMPUTE_TYPE_TURING (Titan RTX)')
    ap.add_argument('--memory', default=None, help='host memory in GB per job (spawner default 8)')
    ap.add_argument('--memory_arm', action='append', default=[], metavar='ARM=GB',
                    help='host memory for one arm, e.g. bl_oops_it1000=32 (OOPS was OOM-killed at 8 GB)')
    ap.add_argument('--relaunch', default=None, help='reason; allows enqueuing cells already in the manifest')
    ap.add_argument('--register-from-index', default=None)
    ap.add_argument('--only', nargs='*', default=None, metavar='TASK:ARM:SEED', help='enqueue only these cells')
    ap.add_argument('--commit', default=None, help='code commit (default HEAD); relaunches keep the batch commit')
    ap.add_argument('--attach_input', action='append', default=[], metavar='DEST=LOCAL_DIR',
                    help='input shipped with each run (spawner --attach_input), e.g. demos not on the cluster')
    ap.add_argument('--pack', action='store_true',
                    help='mark the runs of tasks in PACK packable (supervisor pilots: several runs on one GPU); '
                    'others stay ordinary jobs')
    ap.add_argument('--resumable', action='store_true',
                    help='supervisor continues a stopped run from ckpt/resume.pth (needs code from 328d7b8 on)')
    args = ap.parse_args()

    todo = list(cells(args.batch))
    if args.only:
        want = {tuple(x.split(':')) for x in args.only}
        todo = [c for c in todo if (c[0], c[1], str(c[2])) in want]
    done = {(r['batch'], r['task'], r['arm'], r['seed']) for r in read_manifest()}

    if args.register_from_index:
        # rows of a batch launched by hand: "<tag> <batch> <task> <arm> seed=N run_id commit"
        want = {(t, a, str(s)) for t, a, s in todo}
        n = 0
        for line in open(args.register_from_index):
            m = re.match(rf'^{TAG} {re.escape(args.batch)} (\S+) (\S+) seed=(\d+) (\S+) (\S+)$', line.strip())
            if not m or (m[1], m[2], m[3]) not in want or (args.batch, m[1], m[2], m[3]) in done:
                continue
            append_manifest(dict(batch=args.batch, task=m[1], arm=m[2], seed=m[3], run_id=m[4], commit=m[5],
                                 task_file=f'{TASK_DIR}/{args.batch}/{m[1]}_{m[2]}.yaml',
                                 budget=mtf.budget_of(args.batch, m[1], m[2]), enqueued_utc='', note='registered from run_index'))
            n += 1
        print(f'registered {n} runs of {args.batch}')
        return

    os.chdir(project.CODE)
    if git('status', '--porcelain', 'mineral') and not args.dry_run and not args.commit:
        sys.exit('uncommitted changes in mineral/: commit them first (the stored commit must be the code that ran)')
    # generate task files and check they are committed
    arms = sorted({a for _, a, _ in todo})
    subprocess.run([sys.executable, project.TASK_GENERATOR, args.batch, *arms,
                    '--tasks', *sorted({t for t, _, _ in todo})], check=True)
    if git('status', '--porcelain', f'{TASK_DIR}/{args.batch}') and not args.dry_run:
        sys.exit(f'generated task files of {args.batch} differ from git: commit {TASK_DIR}/{args.batch} first')
    commit = git('rev-parse', args.commit or 'HEAD')

    n = 0
    for task, arm, seed in todo:
        key = (args.batch, task, arm, str(seed))
        if key in done and not args.relaunch:
            continue
        env_file = f'{args.batch}/{task}_{arm}.yaml'
        cmd = ['uv', 'run', '--locked', 'python', 'spawner.py', '--task_name', P['task_name'], '--env_files', env_file,
               '--deployment', 'slurm', '--docker', '--docker_image', IMAGE,
               '--gpu_constraint', args.gpu_constraint, '--supervisor_dir', SUPERVISOR,
               '--no-deploy_now', '--priority', str(args.priority), '--runtime', args.runtime,
               '--git_commit', commit, '--git_branch', P['branch'], '--set', f'seed={seed}']
        if args.gpu_type:
            cmd += ['--gpu_type', args.gpu_type]
        if args.cluster:
            cmd += ['--cluster', args.cluster]
        for a in args.attach_input:
            cmd += ['--attach_input', a]
        if args.pack and task in PACK:
            share, hours, mem_gb = PACK[task]
            cmd += ['--packable', '--gpu_share', str(share), '--gpu_mem_gb', str(mem_gb),
                    '--expected_seconds', str(int(hours * 3600))]
        if args.resumable:
            cmd += ['--resumable']
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
                   budget=mtf.budget_of(args.batch, task, arm), enqueued_utc=datetime.datetime.utcnow().isoformat(timespec='seconds'),
                   note=f'relaunch: {args.relaunch}' if key in done else '')
        append_manifest(row)
        with open(RUN_INDEX, 'a') as f:
            f.write(f'{TAG} {args.batch} {task} {arm} seed={seed} {m[1]} {commit[:7]}\n')
        print(args.batch, task, arm, seed, m[1])
        n += 1
    print(f'{n} runs {"listed" if args.dry_run else "enqueued"} for {args.batch}')


if __name__ == '__main__':
    main()
