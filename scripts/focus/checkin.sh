#!/bin/bash
# One check-in: run states, re-evaluation of newly finished screen runs, screen table, port check, dry-run table.
set -uo pipefail
cd "$(dirname "$0")/../.."
S=/home/joao/docker/cluster-supervisor/data/completed
python3 scripts/focus/status.py | awk '$2 ~ /launched/ {a+=$1} $2=="completed" {c+=$1} $2=="failed" {f+=$1} END{print "active", a, "completed", c, "failed", f}'
python3 scripts/focus/status.py | grep -E "failed" || true
ids=$(awk '$2=="p0screen"||$2=="p2dry"{print $(NF-1)}' ../agent/run_index.tsv | while read r; do [ -f $S/$r.json ] && [ ! -f ../results/data/reeval/$r.json ] && [ -f ~/docker/cluster-supervisor/results/$r/results/ckpt/final.pth ] && awk -v r=$r '$0 ~ r && $2=="p0screen"' ../agent/run_index.tsv | grep -q . && echo $r; done)
[ -n "$ids" ] && TMPDIR=${TMPDIR:-/tmp} timeout 3000 uv run --locked python scripts/focus/reeval.py $ids 2>&1 | grep -E "^[0-9a-f]{32} |Traceback"
python3 scripts/focus/table.py p0screen --expect 3 --allow-incomplete --control ref_otcos 2>&1 | sed -n '5,12p;/### Per seed/,/^$/p' | head -30
python3 scripts/focus/table.py p0screen --expect 3 --allow-incomplete 2>&1 | sed -n '/### Per seed/,/Runs without/p' | head -40
python3 - <<'PY'
import json,os,collections
R=os.path.expanduser('~/docker/cluster-supervisor/results'); E={'hopper':4812,'ant':9318}
out=collections.defaultdict(list)
for l in open('../agent/run_index.tsv'):
    p=l.split()
    if p[1]!='p0port': continue
    f=f'{R}/{p[-2]}/results/scores.json'
    if os.path.exists(f):
        s=json.load(open(f)); r=s['eval_scores/episode_rewards']; out[p[2]].append((p[3][5:], round(sum(r)/len(r)/E[p[2].split('_')[0]],3)))
print('port check (old code, stochastic final):')
for k,v in sorted(out.items()):
    m=sum(x[1] for x in v)/len(v); print(f'  {k:20s} n={len(v)} mean={m:.3f} below0.5={sum(x[1]<0.5 for x in v)}', v)
PY
echo '--- p0screen2'; python3 scripts/focus/table.py p0screen2 --expect 3 --allow-incomplete --control default 2>&1 | sed -n '5,10p'
echo '--- p2dry'; python3 scripts/focus/table.py p2dry --expect 2 --allow-incomplete 2>&1 | sed -n '5,24p'
echo '--- p0screen3'; python3 scripts/focus/table.py p0screen3 --expect 3 --allow-incomplete --control no_minus_c 2>&1 | sed -n '5,10p'
