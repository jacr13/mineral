"""Failure mode of the paper-exact Ant OT-L2 runs (port check): training signals of successful vs failing seeds."""
import csv, collections, os, sys
import numpy as np
R = os.path.expanduser('~/docker/cluster-supervisor/results')
IDX = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../agent/run_index.tsv')
TAGS = ['train_scores/episode_rewards', 'train_scores/episode_lengths', 'train_stats/Lk_min', 'train_stats/Lk_mean',
        'train_stats/pseudo_reward_mean', 'train_stats/pseudo_reward_std', 'train_stats/value_loss',
        'train_stats/grad_norm_before_clip/actor', 'train_stats/entropy_alpha', 'train_stats/avg_kl']
runs = [l.split() for l in open(IDX) if l.startswith('focus p0port ant_otl2 seed')]
marks = [0.25e6, 0.5e6, 1e6, 2e6, 4e6, 7e6, 10e6]
rows = []
for p in runs:
    seed, rid = p[3][5:], p[-2]
    f = os.path.join(R, rid, 'results', 'data.csv')
    if not os.path.exists(f):
        continue
    d = collections.defaultdict(list)
    sig = []
    for row in csv.DictReader(open(f)):
        if row['tag'] in TAGS:
            d[row['tag']].append((float(row['global_step']), float(row['value'])))
        elif row['tag'].startswith('train_stats/sigma'):
            sig.append((float(row['global_step']), float(row['value'])))
    def at(tag, m, w=0.1e6):
        v = [y for x, y in d[tag] if m - w <= x <= m + w]
        return np.mean(v) if v else float('nan')
    final = at('train_scores/episode_rewards', 10e6) / 9329.5
    rows.append((final, seed, {t: [at(t, m) for m in marks] for t in TAGS}))
rows.sort(key=lambda r: -r[0])
print('steps (M):', ' '.join(f'{m/1e6:>7.2f}' for m in marks))
for t in TAGS:
    print('==', t.split('/')[-1] if 'grad' not in t else 'actor_grad_norm')
    for final, seed, v in rows:
        print(f'  seed {seed:>3} final {final:5.2f} |', ' '.join(f'{x:7.3g}' for x in v[t]))
