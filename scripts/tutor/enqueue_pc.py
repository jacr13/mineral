"""Enqueue TUTOR task files on the pc pilot host (Docker image candidj0/mineral:warp13, the image of the clusters'
mineral_uv.sif), one run per seed of the file's sweep block, and register each run in tutor/agent/run_index.tsv.

Refuses a dirty tree for code and a commit that is not on origin/tutor-wip (the supervisor checks out that commit).
usage: python scripts/tutor/enqueue_pc.py <batch> <task_arm.yaml> [...] [--gpu_share 0.3] [--gpu_mem_gb 3]
       [--expected_seconds 1800] [--dry-run]
"""
import argparse
import datetime
import os
import re
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '../..'))
INDEX = os.path.abspath(os.path.join(CODE, '../agent/run_index.tsv'))
SUPERVISOR = '/home/joao/docker/cluster-supervisor/data'
IMAGE = 'candidj0/mineral:warp13'


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True, check=True, cwd=CODE).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--gpu_share', type=float, default=0.3)
    ap.add_argument('--gpu_mem_gb', type=float, default=3.0)
    ap.add_argument('--expected_seconds', type=int, default=1800)
    ap.add_argument('--priority', type=int, default=10)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if git('status', '--porcelain', '--', 'mineral', 'spawner.py', 'tasks/tutor'):
        sys.exit('commit the code and task files first (the run records the commit)')
    head = git('rev-parse', 'HEAD')
    git('fetch', '-q', 'origin', 'tutor-wip')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', head, 'origin/tutor-wip'], cwd=CODE).returncode != 0:
        sys.exit(f'{head[:7]} is not on origin/tutor-wip: push first')
    for f in args.files:
        name = os.path.basename(f)
        body = yaml.safe_load(open(f))
        task = body['task']['env']['env_name']
        arm = name[: -len('.yaml')][len(task) + 1 :]
        cmd = ['uv', 'run', '--locked', 'python', 'spawner.py', '--task_name', f'tutor/gf/{args.batch}', '--env_files', name,
               '--deployment', 'slurm', '--docker', '--container', 'docker', '--docker_image', IMAGE, '--no-deploy_now',
               '--cluster', 'pc', '--supervisor_dir', SUPERVISOR, '--priority', str(args.priority), '--runtime', '12h',
               '--git_branch', 'tutor-wip', '--packable', '--gpu_share', str(args.gpu_share), '--gpu_mem_gb',
               str(args.gpu_mem_gb), '--expected_seconds', str(args.expected_seconds)]
        if args.dry_run:
            print(' '.join(cmd))
            continue
        out = subprocess.run(cmd + ['--enqueue'], capture_output=True, text=True, cwd=CODE)
        ids = re.findall(r'Enqueued run ([0-9a-f]{32})', out.stdout)
        if out.returncode != 0 or len(ids) != len(body['sweep']['seed']):
            sys.exit(f'enqueue failed for {name}:\n{out.stdout[-1500:]}\n{out.stderr[-1500:]}')
        with open(INDEX, 'a') as fh:
            for seed, rid in zip(body['sweep']['seed'], ids):
                fh.write(f'gf {args.batch} {arm} {task}\t{task}\t{seed}\t{head[:7]}\tsupervisor:{rid}\tpc docker {IMAGE}'
                         f'\t{datetime.datetime.now().isoformat(timespec="seconds")}\n')
                print(f'{task} {arm} seed {seed}: {rid}')


if __name__ == '__main__':
    main()
