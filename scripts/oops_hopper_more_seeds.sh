#!/usr/bin/env bash
# Adds the missing seeds (1000, 1100, 1200) to the hopper OOPS `fix_refnet` ablation arm, so hopper has the same
# 6 seeds as the other baselines. The task file is identical to the arm's except for the seed list, so the runs
# land in the SAME wandb group (project oops-fix-ablation-hopper) as the arm's first 3 seeds, and the plotter's
# single group URL covers all 6. 3 jobs.
#
#   ./scripts/oops_hopper_more_seeds.sh              # submit for real
#   DRY_RUN=1 ./scripts/oops_hopper_more_seeds.sh    # only write the sbatch scripts under spawn/oops_ablation/
set -euo pipefail

LOG_FILE="run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

DRY_RUN="${DRY_RUN:-0}"
SLURM_USER="candidor"
POLL_SECONDS=60
MAX_JOBS=100
JOBS_IN_BATCH=3        # seeds in tasks/oops_ablation/dflex_hopper_fix_refnet_more_seeds.yaml

job_count() {
  squeue -h -u "$SLURM_USER" | wc -l | tr -d ' '
}

wait_until_room_for_next_batch() {
  local threshold=$(( MAX_JOBS - JOBS_IN_BATCH ))
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

# Only meaningful with the fixed OOPS.update_net (training used to normalize observations twice, acting only
# once), and it must match the code the first 3 seeds ran. Refuse to submit from a checkout that lacks the fix.
if ! grep -q "normalize twice during training" mineral/agents/oops/oops.py; then
  echo "ERROR: mineral/agents/oops/oops.py does not contain the double-normalization fix." >&2
  echo "       Commit/push it (and git pull on this machine) before launching." >&2
  exit 1
fi

SPAWN_FLAGS=(--no-cleanup --sweep --sweep_max 150)
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: writing sbatch scripts only, nothing will be submitted."
else
  SPAWN_FLAGS+=(--deploy_now)
  wait_until_room_for_next_batch
fi

python spawner.py \
    --task_name oops_ablation \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm \
    "${SPAWN_FLAGS[@]}" \
    --env_files "dflex_hopper_fix_refnet_more_seeds.yaml"
