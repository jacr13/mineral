"""Generate MENTOR task files of the rerun from one design (no task file is edited by hand).

The base of every task is the configuration frozen on 16 Nov (tasks/mentor/icml/dflex_<task>.yaml, D1; HalfCheetah
and CartPole Swing-Up start from Hopper's, see BASE_OF). An arm is the base plus named changes; arms compose with
'-' (``cheap-k16`` = the cheap budget with teacher horizon 16), applied left to right. Every file gets the shared
evaluation protocol (dedicated env, 128 episodes, an evaluation at every round end) and its settings in the body,
never in the sweep block (``--set seed`` replaces the sweep block).

Writes tasks/mentor/icml2/<batch>/<task>_<arm>.yaml in the MENTOR checkout.

usage: python scripts/mentor/make_task_files.py <batch> <arm> [arm ...] [--tasks task ...]   (from the checkout)
"""

import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '../..'))
OUT = 'tasks/mentor/icml2'

# task -> (demonstration file, budget). The budget column of the manifest: 0 = the run's rounds (budget.json says
# whether every round ran); arms with a step cap (capNM) get the cap. The FOCUS budgets (10 M / 15 M; CartPole 4 M)
# are the budget-matched rows.
TASKS = {
    'hopper': ('DFlex_hopper_demos128_return4812_len1000.pt', 0),
    'ant': ('DFlex_ant_demos128_return9318_len1000.pt', 0),
    'humanoid': ('DFlex_humanoid_demos128_return8116_len984.pt', 0),
    'snu_humanoid': ('DFlex_snu_demos128_return6246_len929.pt', 0),
    # not in the cluster's input folder: launch with --attach_input experts/demos=<dir holding this file>
    'cheetah': ('DFlex_cheetah_demos128_return8078_len1000.pt', 0),
    'cartpoleswingup': ('DFlex_cartpoleswingup_demos128_return-286_len240.pt', 0),
}
FOCUS_BUDGET = {'hopper': 10_000_000, 'ant': 10_000_000, 'humanoid': 15_000_000, 'snu_humanoid': 15_000_000,
                'cheetah': 10_000_000, 'cartpoleswingup': 4_000_000}

# frozen task file of each task (16 Nov); new tasks start from Hopper's (same body layout, calibrated gate, K 8)
BASE_OF = {'hopper': 'hopper', 'ant': 'ant', 'humanoid': 'humanoid', 'snu_humanoid': 'snu_humanoid',
           'cheetah': 'hopper', 'cartpoleswingup': 'hopper'}

NO_GATE = {'gate.tau': 1e9, 'gate.quantile': 0, 'gate.auto_noise': 0.0}
# Hopper's calibrated gate (tau = 10 x median teacher cost on noisy 0.2 replays); the calibration sets quantile 0
CAL_GATE = {'gate.tau': 1e9, 'gate.quantile': 0, 'gate.auto_noise': 0.2, 'gate.auto_factor': 10}

# The cheaper budget of X6 (17 Nov, confirmed on 6 old seeds): curriculum selection, 3 teacher batches per round,
# adaptive stop of the teacher's Adam, fewer replay batches; SNU without adaptive stop and with 24 rounds.
_CHEAP = {'teacher.select': 'curriculum', 'teacher.batches_per_round': 3}
CHEAP = {
    'hopper': {**_CHEAP, 'teacher.adaptive_tol': 0.05, 'augment.replay_batches': 1},
    'ant': {**_CHEAP, 'teacher.adaptive_tol': 0.05, 'augment.replay_batches': 1},
    'humanoid': {**_CHEAP, 'teacher.adaptive_tol': 0.02, 'augment.replay_batches': 2},
    'snu_humanoid': {**_CHEAP, 'augment.replay_batches': 2, 'rounds': 24},
    'cheetah': {**_CHEAP, 'teacher.adaptive_tol': 0.05, 'augment.replay_batches': 1},
    'cartpoleswingup': {**_CHEAP, 'teacher.adaptive_tol': 0.05, 'augment.replay_batches': 1},
}

# arm part -> {dotted key under agent.mentor: value}, or a function of the task returning that dict
PARTS = {
    'frozen': {},
    'cheap': lambda task: CHEAP[task],
    'clone': {'rounds': 0},  # Stage I + cloning only
    'nogate': NO_GATE,
    'gatecal': CAL_GATE,
    'r12': {'rounds': 12},
    'r24': {'rounds': 24},
    'r36': {'rounds': 36},
}
for k in (4, 8, 12, 16, 24):
    PARTS[f'k{k}'] = {'teacher.horizon': k}
for n in (1, 2, 4, 8, 16):
    PARTS[f'd{n}'] = {'demos.n_trajs': n}
for m in (4, 10, 15):
    PARTS[f'cap{m}M'] = {'max_sim_steps': m * 1_000_000}


def changes(task, arm):
    out = {}
    for part in arm.split('-'):
        if part not in PARTS:
            raise ValueError(f'{arm}: unknown part {part!r} (known: {sorted(PARTS)})')
        c = PARTS[part]
        out.update(c(task) if callable(c) else c)
    return out


def budget_of(batch, task, arm):
    return int(changes(task, arm).get('max_sim_steps', TASKS[task][1]))


def _set(d, dotted, value):
    keys = dotted.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def task_file(batch, task, arm):
    src = os.path.join(CODE, 'tasks/mentor/icml', f'dflex_{BASE_OF[task]}.yaml')
    base = yaml.safe_load(open(src))
    agent = copy.deepcopy(base['agent'])
    mentor = agent['mentor']
    mentor['demos'] = dict(path='experts/demos/' + TASKS[task][0], n_trajs=8)
    for k, v in changes(task, arm).items():
        _set(mentor, k, v)
    agent = dict(
        name=agent['name'],
        ckpt_every=-1,  # no periodic checkpoints: final.pth (and MENTOR's best-so-far) only
        protocol_eval=True,  # shared evaluation of every method (Agent.protocol_*), on a dedicated env
        eval_every_steps=1,  # MENTOR evaluates at every round end (its policy only changes there) and at step 0
        mentor=mentor,
    )
    return dict(
        task=dict(name='DFlex', env=dict(env_name=task)),
        agent=agent,
        logdir=f'workdir/mentor-{batch}-{task}-{arm}/$(date +%Y%m%d-%H%M%S.%2N)',
        wandb=dict(mode='offline'),
        run='train_eval',
        seed=10,
        sweep=dict(seed=[10, 20, 30]),
        spawner=dict(runtime='12h'),
    )


def main():
    argv = sys.argv[1:]
    tasks = list(TASKS)
    if '--tasks' in argv:
        i = argv.index('--tasks')
        argv, tasks = argv[:i], argv[i + 1:]
    if len(argv) < 2:
        sys.exit(__doc__)
    batch, arms = argv[0], argv[1:]
    out = os.path.join(OUT, batch)
    os.makedirs(out, exist_ok=True)
    for task in tasks:
        for arm in arms:
            body = task_file(batch, task, arm)
            hdr = (f'# Generated by scripts/mentor/make_task_files.py ({batch}, arm {arm}); do not edit.\n'
                   f'# Base: tasks/mentor/icml/dflex_{BASE_OF[task]}.yaml (frozen 16 Nov); changes: '
                   f'{changes(task, arm) or "none"}\n'
                   f'# Settings are in the body, not in the sweep block (--set seed replaces the sweep block).\n')
            with open(os.path.join(out, f'{task}_{arm}.yaml'), 'w') as f:
                f.write(hdr + yaml.safe_dump(body, sort_keys=False, default_flow_style=None))
    print(f'{len(tasks) * len(arms)} task files in {out}')


if __name__ == '__main__':
    main()
