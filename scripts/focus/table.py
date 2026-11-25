"""Tables of the FOCUS rerun, read from the manifest only.

usage (from anywhere):
  python table.py <batch> [--expect 10] [--arms a b ...] [--tasks ...] [--control ARM] [--name NAME]
                  [--allow-incomplete] [--snapshot-only]

For every manifest row of the batch, reads RESULTS_DIR/<run_id>/results/{scores.json,budget.json} (the
supervisor's local copy; fetch missing ones with the supervisor first) and writes the per-seed snapshot
results/data/snapshot_<batch>.tsv, so tables rebuild without the run folders (the snapshot is used for runs whose
files are no longer local).

A cell (task, arm) is written only if it has exactly --expect runs, each deterministic, each at or past its budget,
unless the missing / short run is listed in results/data/exceptions.tsv (batch task arm seed reason). With
--allow-incomplete (screens only) such cells are printed with their count and marked, never written to .tex.

Per cell: mean with bootstrap 95 % interval (10 000 resamples of seeds), IQM (25 % trimmed mean), median, share of
seeds below half the expert, per-seed values, run count, read step. Writes results/tables/<name>.md and, for complete
tables, results/tables/<name>.tex (table body) and results/tables/<name>_values.tex: one definition per number,
used as \\fv{<name>:<task>:<arm>:<stat>}.
"""

import argparse
import json
import os
import sys

import numpy as np

FOCUS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
DATA = os.path.join(FOCUS, 'results/data')
TABLES = os.path.join(FOCUS, 'results/tables')
RESULTS_DIR = os.path.expanduser('~/docker/cluster-supervisor/results')
# Normalisation until 2026-11-25 (Joao, 2026-11-20): mean return of the 8 demonstrations used for training (first 8).
# The 128-episode pool means (4812 / 9318 / 8116 / 6246) include the expert's failed episodes on the humanoids.
# Demo-count arms (1, 2, 4 demos) are normalised by the same 8-demo mean, so they compare directly.
EXPERT = {'hopper': 4813.0, 'ant': 9329.5, 'humanoid': 8225.2, 'snu_humanoid': 6748.9, 'cheetah': 8075.5,
          'jumper': 2073.9}  # Rewarped soft Jumper (Phase 5): first 8 of 64 demonstrations
EXPERT_POOL = {'hopper': 4812.1, 'ant': 9318.1, 'humanoid': 8116.5, 'snu_humanoid': 6246.7, 'cheetah': 8078.6,
               'jumper': 2046.4}
# CartPole Swing-Up (Joao, 2026-11-23): returns are negative and set mostly by the starting state, so the score is
# (R - R_random) / (R_expert - R_random). R_expert: the 128-episode pool mean (random starting states, as the
# evaluation; the first 8 demos had hard starts, -409, below random); R_random: uniform-random actions, 384 episodes
# on the protocol's starting-state seeds (runs 100 / 200 / 300). Other tasks: R / R_expert (R_random = 0).
EXPERT['cartpoleswingup'] = -285.5
EXPERT_POOL['cartpoleswingup'] = -285.5
RANDOM = {'cartpoleswingup': -387.1}


# Reference (Joao, 2026-11-25): the expert's own score, i.e. the mean return of its 128-episode pool (random starting
# states, falls included), not the 8 demonstrations used for training. On SNU the 8 demos are all full-length while
# the expert falls in 44 of 128 episodes, so the 8-demo mean (6749) overstates the expert by 8 %; on the other tasks
# the two agree within 1.4 %. EXPERT (8-demo means) is kept for the record.
REFERENCE = EXPERT_POOL


def norm(task, ret):
    r0 = RANDOM.get(task, 0.0)
    return (ret - r0) / (REFERENCE[task] - r0)


THRESH = 0.9  # 'reached 0.9': first periodic deterministic evaluation at or above 0.9 of the expert
TASK_NAMES = {'hopper': 'Hopper', 'ant': 'Ant', 'humanoid': 'Humanoid', 'snu_humanoid': 'SNU Humanoid',
              'cheetah': 'HalfCheetah', 'jumper': 'Soft Jumper'}
TASK_NAMES['cartpoleswingup'] = 'CartPole Swing-Up'
SNAP_COLS = ['batch', 'task', 'arm', 'seed', 'run_id', 'commit', 'budget', 'agent_steps', 'reached', 'deterministic',
             'episodes', 'return_raw', 'return_norm', 'train_seconds', 'steps_to_09', 'seconds_to_09', 'source']


def time_to_threshold(run_id, task, batch=None):
    """(steps, training seconds) of the first periodic evaluation at or above THRESH x expert, '' if never."""
    f = os.path.join(RESULTS_DIR, run_id, 'results', 'eval_curve.jsonl')
    pts = []
    if os.path.exists(f):
        pts = [json.loads(line) for line in open(f) if line.strip()]
        pts = [(p['agent_steps'], p['train_seconds'], p['mean_return']) for p in pts]
    elif batch is not None:  # run folder released: the batch's curve cache (raw returns), written by curves.py
        pts = [(int(c['agent_steps']), float(c['train_seconds']), float(c['return_raw']))
               for c in read_tsv(os.path.join(DATA, f'curves_{batch}.tsv')) if c['run_id'] == run_id]
    for st, sec, ret in sorted(pts):
        if norm(task, ret) >= THRESH:
            return st, round(sec)
    return ('', '') if pts else (None, None)


def curve_points(run_id, batch=None):
    """(agent steps, raw mean return) of the periodic deterministic evaluations, from the run or the curve cache."""
    f = os.path.join(RESULTS_DIR, run_id, 'results', 'eval_curve.jsonl')
    if os.path.exists(f):
        return sorted((json.loads(line)['agent_steps'], json.loads(line)['mean_return']) for line in open(f) if line.strip())
    if batch is not None:
        return sorted((int(c['agent_steps']), float(c['return_raw']))
                      for c in read_tsv(os.path.join(DATA, f'curves_{batch}.tsv')) if c['run_id'] == run_id)
    return []


def last_k_mean(rec, batch, k=3):
    """Secondary measure (Joao, 2026-11-22): mean of the final score and the k - 1 periodic evaluations before it."""
    pts = [r for st, r in curve_points(rec['run_id'], batch) if st < int(rec['agent_steps'])]
    if len(pts) < k - 1:
        return None
    return (sum(pts[-(k - 1):]) + float(rec['return_raw'])) / k


def read_tsv(path):
    if not os.path.exists(path):
        return []
    lines = open(path).read().splitlines()
    cols = lines[0].split('\t')
    return [dict(zip(cols, line.split('\t'))) for line in lines[1:] if line.strip()]


def write_tsv(path, rows, cols):
    with open(path, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')


def run_record(row, snapshot):
    res = os.path.join(RESULTS_DIR, row['run_id'], 'results')
    rv = os.path.join(DATA, 'reeval', row['run_id'] + '.json')
    if os.path.exists(rv):  # re-evaluated from final.pth with the shared protocol (scripts/focus/reeval.py)
        e = json.load(open(rv))
        rew = e['deterministic']
        bj = os.path.join(res, 'budget.json')
        b = json.load(open(bj)) if os.path.exists(bj) else {}
        steps, budget = int(e['agent_steps']), int(float(row['budget']))
        return dict(
            row, agent_steps=steps, reached=int(steps >= budget and b.get('reached_budget', True)), deterministic=1,
            episodes=len(rew), return_raw=f'{np.mean(rew):.2f}', return_norm=f'{norm(row["task"], np.mean(rew)):.4f}',
            train_seconds=f'{b.get("train_seconds", float("nan")):.0f}', source=f'reeval {e["code_commit"]}')
    sc, bj = os.path.join(res, 'scores.json'), os.path.join(res, 'budget.json')
    if os.path.exists(sc):
        s = json.load(open(sc))
        b = json.load(open(bj)) if os.path.exists(bj) else {}
        if not b and row['run_id'] in snapshot:  # partial download (scores only): keep the recorded training time
            b = dict(train_seconds=float(snapshot[row['run_id']].get('train_seconds') or 'nan'))
        rew = s['eval_scores/episode_rewards']
        steps = int(s['agent_steps'])
        budget = int(float(row['budget']))
        return dict(
            row, agent_steps=steps, reached=int(steps >= budget and b.get('reached_budget', True)),
            deterministic=int(s.get('eval_scores/sample') is False), episodes=len(rew),
            return_raw=f'{np.mean(rew):.2f}', return_norm=f'{norm(row["task"], np.mean(rew)):.4f}',
            train_seconds=f'{b.get("train_seconds", float("nan")):.0f}', source='run files')
    if row['run_id'] in snapshot:
        r = dict(snapshot[row['run_id']], source='snapshot')
        r['return_norm'] = f'{norm(row["task"], float(r["return_raw"])):.4f}'
        return r
    return None


def bootstrap_ci(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def iqm(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    lo, hi = 0.25 * n, 0.75 * n  # fractional trimming, as rliable
    w = np.clip(np.minimum(np.arange(1, n + 1), hi) - np.maximum(np.arange(n), lo), 0, None)
    return float((w * x).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('--expect', type=int, default=10)
    ap.add_argument('--arms', nargs='*')
    ap.add_argument('--tasks', nargs='*')
    ap.add_argument('--control', default=None)
    ap.add_argument('--name', default=None)
    ap.add_argument('--allow-incomplete', action='store_true')
    ap.add_argument('--tex-incomplete', action='store_true',
                    help='tuning batches only (never reported): also write .tex when a cell has fewer runs than '
                         'expected (a documented crash); the n macro of the cell gives its count')
    ap.add_argument('--measure', choices=['final', 'last3'], default='final',
                    help='final: score at the budget (primary); last3: mean of the final score and the two periodic '
                         'evaluations before it (secondary, against single-checkpoint dips)')
    args = ap.parse_args()
    if args.tex_incomplete:
        if 'tune' not in args.batch:
            sys.exit('--tex-incomplete is for tuning batches only')
        args.allow_incomplete = True
    name = args.name or args.batch

    rows = [r for r in read_tsv(os.path.join(DATA, 'manifest.tsv')) if r['batch'] == args.batch]
    # a seed counts once: a relaunch (manifest note 'relaunch: <reason>') replaces the earlier row of the same cell
    latest = {}
    for r in rows:
        latest[(r['task'], r['arm'], r['seed'])] = r
    replaced = [r for r in rows if latest[(r['task'], r['arm'], r['seed'])] is not r]
    rows = list(latest.values())
    if not rows:
        sys.exit(f'no manifest rows for {args.batch}')
    snap_path = os.path.join(DATA, f'snapshot_{args.batch}.tsv')
    snapshot = {r['run_id']: r for r in read_tsv(snap_path)}
    exceptions = {(e['batch'], e['task'], e['arm'], e['seed']): e['reason']
                  for e in read_tsv(os.path.join(DATA, 'exceptions.tsv'))}

    recs, missing = [], []
    for r in rows:
        rec = run_record(r, snapshot)
        if rec is not None:
            st, sec = time_to_threshold(r['run_id'], r['task'], args.batch)
            if st is not None:
                rec['steps_to_09'], rec['seconds_to_09'] = st, sec
        (recs if rec else missing).append(rec or r)
    write_tsv(snap_path, recs, SNAP_COLS)
    if args.measure == 'last3':  # after the snapshot, which always holds the final score
        out = []
        for x in recs:
            m = last_k_mean(x, args.batch)
            if m is None:
                missing.append(x)
                continue
            out.append(dict(x, return_norm=f'{norm(x["task"], m):.4f}'))
        recs = out
        name = f'{name}_last3'
    measure_note = ('' if args.measure == 'final' else
                    ' — secondary measure: mean of the final score and the two periodic evaluations before it')

    tasks = args.tasks or [t for t in EXPERT if any(r['task'] == t for r in rows)]
    arms = args.arms or list(dict.fromkeys(r['arm'] for r in rows))
    lines, values, complete = [], {}, True
    hdr = '| Arm | ' + ' | '.join(TASK_NAMES[t] for t in tasks) + ' |'
    lines += [f'## {name}{measure_note}', '', 'Mean [95 % bootstrap CI] · IQM · share of seeds < 0.5 · n runs @ read step', '',
              hdr, '|:-|' + '-:|' * len(tasks)]
    per_seed = []
    for arm in arms:
        cells = []
        for t in tasks:
            exp_rows = [r for r in rows if r['task'] == t and r['arm'] == arm]
            if not exp_rows:
                cells.append('–')
                continue
            got = [x for x in recs if x['task'] == t and x['arm'] == arm]
            # an exception excuses the budget rule for that run (it is reported, and the reason is printed);
            # a run that is missing or not deterministic is never counted
            exc = lambda x: (args.batch, t, arm, x['seed']) in exceptions
            ok = [x for x in got if (int(x['reached']) or exc(x)) and int(x['deterministic']) and int(x['episodes']) == 128]
            bad = [r for r in exp_rows if r['run_id'] not in {x['run_id'] for x in ok}]
            unexcused = bad
            n_ok = len(ok)
            cell_complete = not unexcused and n_ok == args.expect and len(exp_rows) == args.expect
            if not cell_complete:
                complete = False
                if not args.allow_incomplete:
                    status = ', '.join(f'seed {r["seed"]}' for r in unexcused)
                    sys.exit(f'refusing {t}/{arm}: {n_ok} of {args.expect} runs complete; not complete: {status}')
            if n_ok == 0:
                cells.append(f'(0/{len(exp_rows)})')
                continue
            v = np.array([float(x['return_norm']) for x in ok])
            lo, hi = bootstrap_ci(v)
            step = min(int(x['agent_steps']) for x in ok)
            st = dict(mean=v.mean(), lo=lo, hi=hi, iqm=iqm(v), median=float(np.median(v)), fail=float((v < 0.5).mean()),
                      n=n_ok, step=step)
            values[(t, arm)] = st
            mark = '' if cell_complete else ' **(incomplete)**'
            cells.append(f'{st["mean"]:.3f} [{lo:.3f}, {hi:.3f}] · {st["iqm"]:.3f} · {st["fail"]:.0%} · '
                         f'n={n_ok} @ {step / 1e6:.1f}M{mark}')
            per_seed.append(f'| {arm} | {TASK_NAMES[t]} | ' + ', '.join(
                f'{x["seed"]}: {float(x["return_norm"]):.3f}' for x in sorted(ok, key=lambda x: int(x['seed']))) + ' |')
        lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    if args.control:
        lines += ['', f'Difference to `{args.control}` (mean, same batch):', '']
        lines.append(hdr)
        lines.append('|:-|' + '-:|' * len(tasks))
        for arm in arms:
            if arm == args.control:
                continue
            c = [f'{values[(t, arm)]["mean"] - values[(t, args.control)]["mean"]:+.3f}'
                 if (t, arm) in values and (t, args.control) in values else '–' for t in tasks]
            lines.append(f'| {arm} | ' + ' | '.join(c) + ' |')
    lines += ['', f'Reaching {THRESH} of the expert (first periodic deterministic evaluation): runs reaching / runs · '
              'median steps (M) · median training time (h, evaluation excluded), over the runs that reach it', '',
              hdr, '|:-|' + '-:|' * len(tasks)]
    for arm in arms:
        cells = []
        for t in tasks:
            got = [x for x in recs if x['task'] == t and x['arm'] == arm and x.get('steps_to_09') not in (None,)]
            known = [x for x in got if 'steps_to_09' in x]
            hit = [x for x in known if str(x['steps_to_09']) != '']
            if not known:
                cells.append('–')
                continue
            if not hit:
                cells.append(f'0/{len(known)}')
                continue
            ms = np.median([float(x['steps_to_09']) for x in hit]) / 1e6
            mh = np.median([float(x['seconds_to_09']) for x in hit]) / 3600
            cells.append(f'{len(hit)}/{len(known)} · {ms:.1f} M · {mh:.2f} h')
        lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    lines += ['', '### Per seed', '', '| Arm | Task | seed: return / expert |', '|:-|:-|:-|'] + per_seed
    if missing:
        lines += ['', f'Runs without local files or snapshot ({len(missing)}): ' +
                  ', '.join(f'{r["task"]}/{r["arm"]}/{r["seed"]}' for r in missing)]
    if replaced:
        lines += ['', f'Relaunched (the latest launch of each seed is used): ' +
                  ', '.join(f'{r["task"]}/{r["arm"]}/{r["seed"]}' for r in replaced)]
    if exceptions:
        lines += ['', 'Exceptions applied: ' + '; '.join(f'{k[1]}/{k[2]}/{k[3]}: {v}' for k, v in exceptions.items()
                                                       if k[0] == args.batch)]
    os.makedirs(TABLES, exist_ok=True)
    md = '\n'.join(lines) + '\n'
    open(os.path.join(TABLES, f'{name}.md'), 'w').write(md)
    print(md)
    if complete or args.tex_incomplete:
        with open(os.path.join(TABLES, f'{name}_values.tex'), 'w') as f:
            f.write(f'% generated by scripts/focus/table.py {args.batch}; one definition per number\n')
            f.write('\\providecommand{\\fv}[1]{\\csname fv@#1\\endcsname}\n')
            for (t, arm), st in values.items():
                for k in ('mean', 'lo', 'hi', 'iqm', 'median', 'fail', 'n', 'step'):
                    val = (f'{st[k]:.3f}' if k not in ('n', 'step', 'fail') else
                           f'{st[k]}' if k == 'n' else f'{st[k] / 1e6:.1f}M' if k == 'step' else f'{st[k]:.0%}'.replace('%', '\\%'))
                    f.write(f'\\expandafter\\def\\csname fv@{name}:{t}:{arm}:{k}\\endcsname{{{val}}}\n')
        with open(os.path.join(TABLES, f'{name}.tex'), 'w') as f:
            f.write(f'% generated by scripts/focus/table.py {args.batch}; \\input after {name}_values.tex\n')
            for arm in arms:
                f.write(arm.replace('_', '\\_') + ' & ' + ' & '.join(
                    f'\\fv{{{name}:{t}:{arm}:mean}} {{\\scriptsize [\\fv{{{name}:{t}:{arm}:lo}}, '
                    f'\\fv{{{name}:{t}:{arm}:hi}}]}}' if (t, arm) in values else '--' for t in tasks) + ' \\\\\n')
            f.write(f'% n runs per cell and read step: \\fv{{{name}:<task>:<arm>:n}}, \\fv{{{name}:<task>:<arm>:step}}\n')
        print(f'wrote {name}.tex and {name}_values.tex')
    else:
        print('incomplete: no .tex written')


if __name__ == '__main__':
    main()
