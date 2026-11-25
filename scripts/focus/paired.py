"""Paired comparisons, the rerun noise floor and the aggregate over tasks, from the batch snapshots.

usage: python scripts/focus/paired.py            (reads results/data/snapshot_{p3main,p4zo}.tsv)

Writes results/tables/paired.{md,tex} and paired_values.tex (writer requests R2, R3, R6, 2026-11-25).

Paired difference: for an arm A and a reference B on one task, d_s = score_A(s) - score_B(s) over the seeds s both
have (normally the 10 reported seeds); reported as mean d with a 95 % bootstrap interval (10 000 resamples of seeds),
the number of seeds where A > B, and a flag when the interval excludes 0. Scores are table.norm (the expert's
128-episode mean). Cells with fewer than 10 paired seeds are skipped, never printed.

Sets (R2): every ablation arm vs f_main_ot (4 tasks); f_main_dtw, f_main_noalign vs f_main_ot (6 tasks); f_main_ot
vs every baseline (6 tasks); f_ab_demos1 vs every baseline (4 tasks); every noisy arm vs its clean control;
f_zo_ppo, f_zo_sac vs the p4zo control. R3: the p4zo control vs the p3main control (same configuration and seeds,
other run / GPU), the noise floor of "the same configuration run twice".

Aggregate (R6): IQM over task x seed per method, with a stratified bootstrap interval (seeds resampled within each
task, 10 000 times), over the six tasks and over Ant + Humanoid + SNU.

Macros: \\fv{paired:<set>:<task>:<arm>:<ref>:<field>} with field diff, lo, hi, won, n, sig (1/0);
\\fv{agg:<tasks>:<arm>:<field>} with field iqm, lo, hi.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import table  # noqa: E402

TABLES = table.TABLES
TASKS6 = ['hopper', 'ant', 'humanoid', 'snu_humanoid', 'cheetah', 'cartpoleswingup']
TASKS4 = TASKS6[:4]
HARD = ['ant', 'humanoid', 'snu_humanoid']
BASELINES = ['bl_gaifo_tuned', 'bl_maad_tuned', 'bl_dac', 'bl_opolo', 'bl_pwil_tuned', 'bl_lwail', 'bl_oops_tuned',
             'bl_ild_tuned', 'bl_diffmimic', 'bl_bco', 'bl_gail_tuned', 'bl_bc']
NB = 10_000


def load(batch):
    out = {}
    with open(os.path.join(table.DATA, f'snapshot_{batch}.tsv')) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            out.setdefault((r['task'], r['arm']), {})[int(r['seed'])] = table.norm(r['task'], float(r['return_raw']))
    return out


def paired(a, b, seed=0):
    k = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in k])
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (NB, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return dict(diff=d.mean(), lo=lo, hi=hi, won=int((d > 0).sum()), n=len(d), sig=int(lo > 0 or hi < 0))


def iqm(x):
    x = np.sort(x, axis=-1)
    n = x.shape[-1]
    k = int(np.floor(0.25 * n))
    return x[..., k:n - k].mean(-1)


def aggregate(data, arm, tasks, seed=0):
    per = [np.array(list(data[(t, arm)].values())) for t in tasks]
    if any(len(p) < 10 for p in per):
        return None
    rng = np.random.default_rng(seed)
    boot = np.concatenate([p[rng.integers(0, len(p), (NB, len(p)))] for p in per], axis=1)
    lo, hi = np.percentile(iqm(boot), [2.5, 97.5])
    return dict(iqm=iqm(np.concatenate(per)), lo=lo, hi=hi)


def main():
    main_ = load('p3main')
    zo = load('p4zo')
    comps = []  # (set, task, arm, ref, data_arm, data_ref)
    abl = sorted({a for _, a in main_ if a.startswith('f_ab_')})
    for arm in abl:
        for t in TASKS4:
            comps.append(('ablation', t, arm, 'f_main_ot', main_.get((t, arm)), main_.get((t, 'f_main_ot'))))
    for arm in ('f_main_dtw', 'f_main_noalign'):
        for t in TASKS6:
            comps.append(('variant', t, arm, 'f_main_ot', main_.get((t, arm)), main_.get((t, 'f_main_ot'))))
    for bl in BASELINES:
        for t in TASKS6:
            comps.append(('vsbaseline', t, 'f_main_ot', bl, main_.get((t, 'f_main_ot')), main_.get((t, bl))))
        for t in TASKS4:
            comps.append(('demos1vsbaseline', t, 'f_ab_demos1', bl, main_.get((t, 'f_ab_demos1')), main_.get((t, bl))))
    for (t, arm) in sorted(main_):
        if '__n' in arm:
            comps.append(('noisy', t, arm, arm.split('__n')[0], main_.get((t, arm)), main_.get((t, arm.split('__n')[0]))))
    for arm in ('f_zo_ppo', 'f_zo_sac'):
        for t in TASKS6:
            comps.append(('zeroorder', t, arm, 'f_main_ot', zo.get((t, arm)), zo.get((t, 'f_main_ot'))))
    for t in TASKS6:  # R3: same configuration and seeds, run twice (p4zo control vs p3main control)
        comps.append(('rerun', t, 'f_main_ot@p4zo', 'f_main_ot@p3main', zo.get((t, 'f_main_ot')), main_.get((t, 'f_main_ot'))))

    vals, md, tex = {}, [], []
    cur = None
    for i, (st, t, arm, ref, xa, xb) in enumerate(comps):
        if not xa or not xb or len(set(xa) & set(xb)) < 10:
            continue
        p = paired(xa, xb, seed=i)
        key = f'paired:{st}:{t}:{arm}:{ref}'
        for f, v in p.items():
            vals[f'{key}:{f}'] = f'{v:+.3f}' if f in ('diff', 'lo', 'hi') else str(v)
        if st != cur:
            md += ['', f'### {st}', '', '| Task | Arm | Reference | Δ [95 % CI] | won | sig |', '|:-|:-|:-|-:|-:|:-|']
            cur = st
        md.append(f'| {table.TASK_NAMES.get(t, t)} | {arm} | {ref} | {p["diff"]:+.3f} [{p["lo"]:+.3f}, {p["hi"]:+.3f}] '
                  f'| {p["won"]}/{p["n"]} | {"*" if p["sig"] else ""} |')
        e = lambda s: s.replace('_', '\\_')  # noqa: E731
        tex.append(f'{e(st)} & {e(t)} & {e(arm)} & {e(ref)} & \\fv{{{key}:diff}} {{\\scriptsize [\\fv{{{key}:lo}}, '
                   f'\\fv{{{key}:hi}}]}} & \\fv{{{key}:won}}/\\fv{{{key}:n}} \\\\')

    md += ['', '### Aggregate: IQM over task x seed, stratified bootstrap 95 % CI', '',
           '| Method | six tasks | Ant + Humanoid + SNU |', '|:-|-:|-:|']
    for arm in ['f_main_ot', 'f_main_dtw', 'f_main_noalign'] + BASELINES:
        cells = []
        for name, tasks in (('all6', TASKS6), ('hard3', HARD)):
            g = aggregate(main_, arm, tasks)
            if g is None:
                cells.append('–')
                continue
            for f, v in g.items():
                vals[f'agg:{name}:{arm}:{f}'] = f'{v:.3f}'
            cells.append(f'{g["iqm"]:.3f} [{g["lo"]:.3f}, {g["hi"]:.3f}]')
        md.append(f'| {arm} | ' + ' | '.join(cells) + ' |')

    head = ('Paired differences (arm − reference, same seeds), normalised by the expert 128-episode mean; '
            '* = 95 % bootstrap interval excludes 0. Generated by scripts/focus/paired.py.')
    with open(os.path.join(TABLES, 'paired.md'), 'w') as f:
        f.write(head + '\n' + '\n'.join(md) + '\n')
    with open(os.path.join(TABLES, 'paired_values.tex'), 'w') as f:
        f.write('% generated by scripts/focus/paired.py; one definition per number\n')
        f.write('\\providecommand{\\fv}[1]{\\csname fv@#1\\endcsname}\n')
        for k, v in vals.items():
            f.write(f'\\expandafter\\def\\csname fv@{k}\\endcsname{{{v}}}\n')
    with open(os.path.join(TABLES, 'paired.tex'), 'w') as f:
        f.write('% generated by scripts/focus/paired.py; \\input after paired_values.tex\n')
        f.write('% columns: set & task & arm & reference & difference [95 % CI] & seeds won\n')
        f.write('\n'.join(tex) + '\n')
    print(f'{sum(1 for k in vals if k.endswith(":diff"))} paired cells, '
          f'{sum(1 for k in vals if k.startswith("agg:") and k.endswith(":iqm"))} aggregates')


if __name__ == '__main__':
    main()
