#!/usr/bin/env bash
# Launches the OOPS hopper ablation (tasks/oops_ablation/*.yaml) on the SLURM cluster:
# one task file per arm, 3 seeds each (9 jobs total). Runtime, GPU constraint, wandb project
# and seeds live in each task file, so nothing is overridden here.
#
#   ./scripts/oops_ablation.sh              # submit for real
#   DRY_RUN=1 ./scripts/oops_ablation.sh    # only write the sbatch scripts under spawn/oops_ablation/
set -euo pipefail

LOG_FILE="run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

DRY_RUN="${DRY_RUN:-0}"
SLURM_USER="candidor"
POLL_SECONDS=60
MAX_JOBS=100
JOBS_PER_ARM=3         # one job per seed in each task file's `sweep.seed`

ARMS=(
  "dflex_hopper_fix"              # fix only (default small networks, 8 demos)
  "dflex_hopper_fix_refnet"       # + reference-sized networks
  "dflex_hopper_fix_refnet_1demo" # + a single expert demo
)

job_count() {
  squeue -h -u "$SLURM_USER" | wc -l | tr -d ' '
}

wait_until_room_for_next_batch() {
  local threshold=$(( MAX_JOBS - JOBS_PER_ARM ))
  if [[ "$threshold" -lt 0 ]]; then
    threshold=0
  fi

  while true; do
    local n
    n="$(job_count)"
    if [[ "$n" -le "$threshold" ]]; then
      echo "Queue size ${n} <= threshold ${threshold}. Room for another batch."
      break
    fi
    echo "Queue size ${n} > ${threshold}. Sleeping ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
  done
}

if [[ "$DRY_RUN" != "1" ]]; then
  git pull
fi

# The ablation is only meaningful with the fixed OOPS.update_net (training used to normalize
# observations twice, acting only once). Refuse to submit from a checkout that lacks the fix,
# e.g. if it has not been pushed to this cluster's repo yet.
if ! grep -q "normalize twice during training" mineral/agents/oops/oops.py; then
  echo "ERROR: mineral/agents/oops/oops.py does not contain the double-normalization fix." >&2
  echo "       Commit/push it (and git pull on this machine) before launching the ablation." >&2
  exit 1
fi

SPAWN_FLAGS=(--no-cleanup --sweep --sweep_max 150)
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: writing sbatch scripts only, nothing will be submitted."
else
  SPAWN_FLAGS+=(--deploy_now)
fi

for arm in "${ARMS[@]}"; do
    echo "============================================================"
    echo "Launching OOPS ablation arm: ${arm}"
    echo "============================================================"

    if [[ "$DRY_RUN" != "1" ]]; then
        wait_until_room_for_next_batch
    fi

    python spawner.py \
        --task_name oops_ablation \
        --docker \
        --docker_image /home/users/c/candidor/docker/mineral.sif \
        --deployment slurm \
        "${SPAWN_FLAGS[@]}" \
        --env_files "${arm}.yaml"
done
