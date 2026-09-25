#!/usr/bin/env bash
# Launches only OOPS hopper: 6 seeds, 12h each (tasks/oops_refnet/dflex_hopper.yaml).
set -euo pipefail

git pull

if ! grep -q "normalize twice during training" mineral/agents/oops/oops.py; then
  echo "ERROR: mineral/agents/oops/oops.py lacks the double-normalization fix; push it and git pull first." >&2
  exit 1
fi

python spawner.py \
    --task_name oops_refnet \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm \
    --no-cleanup \
    --sweep \
    --sweep_max 150 \
    --deploy_now \
    --env_files dflex_hopper.yaml
