"""Wall-clock of every Phase 3 method on one GPU model (RTX 3090), extrapolated from the timing batch.

usage: python scripts/focus/timing.py [--timing p3time] [--main p3main]

For each method (arm) and task:
  t5      = training seconds of the RTX 3090 timing run (seed 100, 5 % of the budget: make_task_files.BUDGET_SCALE),
            evaluation excluded, over s5 agent steps.
  ratio   = for each full run of the same arm in the main batch (any GPU): training seconds at the budget divided by
            training seconds at s5 steps (linear interpolation between periodic evaluations). It is the method's own
            time profile (warm-up phases, replay growth), measured on its own runs.
  full    = t5 x median(ratio); the spread is t5 x the quartiles of ratio.
  to 0.9  = for runs that reach 0.9 of the expert (first periodic deterministic evaluation): t5 x T(s_0.9) / T(s5),
            median over those runs.
Methods whose timing run already covers the full budget (BC, BCO: their budget is not scaled) use t5 directly.
Reads the curve cache results/data/curves_<main>.tsv and the run folders of the timing batch; writes
results/data/timing_<timing>.tsv (the timing runs, so the table rebuilds after release) and prints the table.
"""
import argparse, csv, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FOCUS = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(FOCUS, 'results', 'data')
RES = os.path.expanduser('~/docker/cluster-supervisor/results')
sys.path.insert(0, HERE)
import table  # noqa: E402  (EXPERT normalisation)

TASKS = ['hopper', 'ant', 'humanoid', 'snu_humanoid', 'cheetah', 'cartpoleswingup']


def manifest(batch):
    latest = {}
    with open(os.path.join(DATA, 'manifest.tsv')) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['batch'] == batch:
                latest[(r['task'], r['arm'], r['seed'])] = r
    return latest


def timing_runs(batch):
    path = os.path.join(DATA, f'timing_{batch}.tsv')
    cached = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f, delimiter='\t'):
                cached[r['run_id']] = r
    out = {}
    for (task, arm, seed), r in manifest(batch).items():
        b = os.path.join(RES, r['run_id'], 'results', 'budget.json')
        if r['run_id'] in cached:
            c = cached[r['run_id']]
        elif os.path.exists(b):
            j = json.load(open(b))
            c = dict(run_id=r['run_id'], task=task, arm=arm, seed=seed, agent_steps=j['agent_steps'],
                     train_seconds=j['train_seconds'], reached_budget=j['reached_budget'])
        else:
            continue
        cached[r['run_id']] = c
        out[(task, arm)] = c
    with open(path, 'w') as f:
        w = csv.DictWriter(f, delimiter='\t', fieldnames=['run_id', 'task', 'arm', 'seed', 'agent_steps',
                                                          'train_seconds', 'reached_budget'])
        w.writeheader()
        for c in cached.values():
            w.writerow(c)
    return out


def curves(batch):
    by_run = {}
    with open(os.path.join(DATA, f'curves_{batch}.tsv')) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            by_run.setdefault(r['run_id'], []).append((int(r['agent_steps']), float(r['train_seconds']),
                                                       float(r['return_raw'])))
    return {k: sorted(v) for k, v in by_run.items()}


def t_at(curve, steps):
    for (s0, t0, _), (s1, t1, _) in zip(curve, curve[1:]):
        if s0 <= steps <= s1:
            return t0 + (t1 - t0) * (steps - s0) / max(1, s1 - s0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', default='p3time')
    ap.add_argument('--main', default='p3main')
    a = ap.parse_args()
    tim, cur, main_runs = timing_runs(a.timing), curves(a.main), manifest(a.main)
    arms = sorted({arm for _, arm in tim}, key=lambda x: (not x.startswith('f_'), x))
    rows = []
    for arm in arms:
        cells = []
        for task in TASKS:
            c = tim.get((task, arm))
            if c is None:
                cells.append('–')
                continue
            t5, s5 = float(c['train_seconds']), int(c['agent_steps'])
            ratios, to90 = [], []
            full_steps = None
            for (t, ar, _), r in main_runs.items():
                if t != task or ar != arm or r['run_id'] not in cur:
                    continue
                cv = cur[r['run_id']]
                full_steps = cv[-1][0]
                if s5 >= full_steps:  # the timing run covers the whole budget
                    continue
                ts5 = t_at(cv, s5)
                if not ts5:
                    continue
                ratios.append(cv[-1][1] / ts5)
                hit = next(((s, tt) for s, tt, ret in cv if table.norm(task, ret) >= 0.9), None)
                if hit:
                    to90.append(t5 * hit[1] / ts5)
            if full_steps is not None and s5 >= full_steps:
                cells.append(f'{t5 / 3600:.2f} h (measured)')
                continue
            if not ratios:
                cells.append(f'5 %: {t5 / 60:.0f} min, no full run yet')
                continue
            q = statistics.quantiles(ratios, n=4) if len(ratios) >= 2 else [ratios[0]] * 3
            cell = f'{t5 * q[1] / 3600:.2f} h [{t5 * q[0] / 3600:.2f}, {t5 * q[2] / 3600:.2f}] n={len(ratios)}'
            if to90:
                cell += f' · 0.9: {statistics.median(to90) / 3600:.2f} h ({len(to90)}/{len(ratios)})'
            cells.append(cell)
        rows.append((arm, cells))
    print(f'## Wall-clock on RTX 3090 ({a.timing} timing runs, extrapolated with the {a.main} time profiles)\n')
    print('Training hours at the budget, median [quartiles] over the full runs\' profiles; time to 0.9 of the expert '
          '(median over the runs that reach it). Evaluation time excluded.\n')
    print('| Arm | ' + ' | '.join(table.TASK_NAMES.get(t, t) if hasattr(table, 'TASK_NAMES') else t
                                 for t in TASKS) + ' |')
    print('|:-|' + '-:|' * len(TASKS))
    for arm, cells in rows:
        print(f'| {arm} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
