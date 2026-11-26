#!/usr/bin/env bash
# One local TUTOR run (GTX 1070; nothing is sent to the cluster): frozen MENTOR configuration of <task> with the
# gradient-free Stage I (scripts/tutor/overrides.py), the shared evaluation protocol (128 deterministic episodes on
# a dedicated env, an evaluation at every round end), plus <key=val> changes under agent.mentor.
# Appends one line to tutor/agent/run_index.tsv: tag, task, seed, commit, logdir, changes, start time.
# usage: scripts/tutor/local_run.sh "<tag starting with gf>" <task> <seed> [key=val ...]
set -euo pipefail
tag=$1 task=$2 seed=$3; shift 3
cd "$(dirname "$0")/../.."
root=$(cd ../ && pwd)
name=$(echo "$tag" | tr ' ' '_')
logdir=$root/results/local_runs/${name}_s${seed}_$(date +%Y%m%d-%H%M%S)
commit=$(git rev-parse --short HEAD)$(git diff --quiet -- mineral || echo "-dirty")
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$task" "$seed" "$commit" "$logdir" "$*" "$(date -Is)" >> "$root/agent/run_index.tsv"
ov=$(uv run --locked python scripts/tutor/overrides.py "$task" "$logdir" "$seed" "$@" | sed "s/ run=train / /")
mkdir -p "$logdir"
uv run --locked python -m mineral.scripts.run $ov run=train_eval agent.protocol_eval=True agent.eval_every_steps=1 \
  agent.ckpt_every=-1 > "$logdir/stdout.log" 2>&1
echo "done $logdir"
