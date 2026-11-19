"""Learning curves of a batch from the periodic deterministic evaluations (eval_curve.jsonl), read via the manifest.

usage: python curves.py <batch> [--arms ...] [--tasks ...] [--name NAME]

Per (task, arm): the return / expert of each run at each evaluation step; mean and 95 % bootstrap band over the runs
that have that step. A curve ends at the last step every run of the cell reached: no extension, no holding the last
value. Also against training time (seconds, evaluation excluded), on a common time grid ending at the shortest run.
Writes results/figures/<name>_{steps,time}.{pdf,png} and results/data/curves_<batch>.tsv (per run, per point), so the
figures rebuild without the run folders.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from table import DATA, EXPERT, FOCUS, RESULTS_DIR, TASK_NAMES, bootstrap_ci, read_tsv, write_tsv  # noqa: E402

FIG = os.path.join(FOCUS, 'results/figures')


def load_curves(rows, cache):
    out = {}
    for r in rows:
        f = os.path.join(RESULTS_DIR, r['run_id'], 'results', 'eval_curve.jsonl')
        if os.path.exists(f):
            pts = [json.loads(line) for line in open(f) if line.strip()]
            out[r['run_id']] = [(p['agent_steps'], p['train_seconds'], p['mean_return'] / EXPERT[r['task']]) for p in pts]
        elif r['run_id'] in cache:
            out[r['run_id']] = cache[r['run_id']]
    return out


def band(ys):
    ys = np.asarray(ys, float)
    lo, hi = bootstrap_ci(ys, n=2000) if len(ys) > 1 else (ys[0], ys[0])
    return ys.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('--arms', nargs='*')
    ap.add_argument('--tasks', nargs='*')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    name = args.name or args.batch
    rows = [r for r in read_tsv(os.path.join(DATA, 'manifest.tsv')) if r['batch'] == args.batch]
    cache_path = os.path.join(DATA, f'curves_{args.batch}.tsv')
    cache = {}
    for c in read_tsv(cache_path):
        cache.setdefault(c['run_id'], []).append((int(c['agent_steps']), float(c['train_seconds']), float(c['return_norm'])))
    curves = load_curves(rows, cache)
    write_tsv(cache_path, [dict(run_id=k, agent_steps=s, train_seconds=f'{t:.1f}', return_norm=f'{y:.4f}')
                           for k, pts in curves.items() for s, t, y in pts],
              ['run_id', 'agent_steps', 'train_seconds', 'return_norm'])
    tasks = args.tasks or [t for t in EXPERT if any(r['task'] == t for r in rows)]
    arms = args.arms or list(dict.fromkeys(r['arm'] for r in rows))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    os.makedirs(FIG, exist_ok=True)
    for axis in ('steps', 'time'):
        fig, axs = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.0), squeeze=False)
        for ax, t in zip(axs[0], tasks):
            for i, arm in enumerate(arms):
                runs = [curves[r['run_id']] for r in rows if r['task'] == t and r['arm'] == arm and r['run_id'] in curves]
                if not runs:
                    continue
                if axis == 'steps':
                    common = sorted(set.intersection(*[{s for s, _, _ in c} for c in runs]))
                    xs = np.array(common, float)
                    ys = [[dict((s, y) for s, _, y in c)[x] for c in runs] for x in common]
                    xplot = xs / 1e6
                else:
                    tmax = min(c[-1][1] for c in runs)
                    xs = np.linspace(0, tmax, 50)
                    ys = [[np.interp(x, [p[1] for p in c], [p[2] for p in c]) for c in runs] for x in xs]
                    xplot = xs / 3600
                m = np.array([band(y) for y in ys])
                ax.plot(xplot, m[:, 0], color=colors[i % len(colors)], label=f'{arm} (n={len(runs)})', lw=1.5)
                ax.fill_between(xplot, m[:, 1], m[:, 2], color=colors[i % len(colors)], alpha=0.2, lw=0)
            ax.set_title(TASK_NAMES[t])
            ax.set_xlabel('environment steps (M)' if axis == 'steps' else 'training time (h)')
            ax.set_ylim(-0.05, 1.1)
            ax.grid(alpha=0.3)
        axs[0][0].set_ylabel('return / expert (deterministic eval)')
        axs[0][-1].legend(fontsize=7, loc='lower right')
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(FIG, f'{name}_{axis}.{ext}'), dpi=150)
        plt.close(fig)
    print(f'wrote {FIG}/{name}_{{steps,time}}.{{pdf,png}}')


if __name__ == '__main__':
    main()
