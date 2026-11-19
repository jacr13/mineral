"""Status of this agent's runs: supervisor state per tag, and the latest periodic evaluation of runs with local files.

usage: python status.py [tag-substring]
"""
import collections, json, os, sys
SUP = os.path.expanduser('~/docker/cluster-supervisor/data')
RES = os.path.expanduser('~/docker/cluster-supervisor/results')
IDX = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../agent/run_index.tsv')
E = {'hopper': 4812, 'ant': 9318, 'humanoid': 8116, 'snu_humanoid': 6246}
pat = sys.argv[1] if len(sys.argv) > 1 else ''
cnt, latest = collections.Counter(), []
for line in open(IDX):
    p = line.split()
    if pat not in line or len(p) < 4:
        continue
    rid, tag = p[-2], ' '.join(p[1:-3])
    st = 'unknown'
    for d in ('pending', 'submitting', 'launched', 'completed', 'failed', 'cancelled'):
        f = os.path.join(SUP, d, rid + '.json')
        if os.path.exists(f):
            st = d
            if d == 'launched':
                st = 'launched/' + str(json.load(open(f)).get('slurm_state'))
            break
    cnt[(tag, st)] += 1
    cur = os.path.join(RES, rid, 'results', 'eval_curve.jsonl')
    sc = os.path.join(RES, rid, 'results', 'scores.json')
    task = next((t for t in sorted(E, key=len, reverse=True) if t in tag.replace(' ', '_') or tag.startswith(t)), None)
    if os.path.exists(cur) and task:
        pts = [json.loads(l) for l in open(cur) if l.strip()]
        if pts:
            fin = ''
            if os.path.exists(sc):
                s = json.load(open(sc)); r = s['eval_scores/episode_rewards']
                fin = f' final {sum(r) / len(r) / E[task]:.3f} (sample={s.get("eval_scores/sample")})'
            latest.append(f'{tag} {p[-3]}: {pts[-1]["agent_steps"] / 1e6:.1f}M -> {pts[-1]["mean_return"] / E[task]:.3f}, '
                          f'{pts[-1]["train_seconds"] / 3600:.2f} h{fin}')
for (tag, st), n in sorted(cnt.items()):
    print(f'{n:3d} {st:22s} {tag}')
print('\n'.join(latest))
