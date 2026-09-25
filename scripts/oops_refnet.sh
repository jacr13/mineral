#!/usr/bin/env bash
# Launches OOPS with the configuration that learned in the hopper ablation (tasks/oops_refnet/*.yaml:
# fixed observation normalization + reference-sized networks), 6 seeds per environment.
# Runtime, GPU constraint, wandb project and seeds live in each task file.
#
#   ./scripts/oops_refnet.sh                            # ant, humanoid, snu_humanoid
#   (hopper is not in the default set: its 3 ablation seeds are completed with scripts/oops_hopper_more_seeds.sh;
#    `ENVS=hopper` here instead launches a fresh, separate 6-seed hopper group)
#   ENVS="hopper ant" ./scripts/oops_refnet.sh          # choose environments
#   DRY_RUN=1 ./scripts/oops_refnet.sh                  # only write the sbatch scripts under spawn/oops_refnet/
set -euo pipefail

LOG_FILE="run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

DRY_RUN="${DRY_RUN:-0}"
SLURM_USER="candidor"
POLL_SECONDS=60
MAX_JOBS=100
JOBS_PER_ENV=6         # one job per seed in each task file's `sweep.seed`

read -r -a ENV_LIST <<< "${ENVS:-ant humanoid snu_humanoid}"

job_count() {
  squeue -h -u "$SLURM_USER" | wc -l | tr -d ' '
}

wait_until_room_for_next_batch() {
  local threshold=$(( MAX_JOBS - JOBS_PER_ENV ))
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

# These runs are only meaningful with the fixed OOPS.update_net (training used to normalize
# observations twice, acting only once). Refuse to submit from a checkout that lacks the fix,
# e.g. if it has not been pushed to this cluster's repo yet.
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
fi

for env in "${ENV_LIST[@]}"; do
    if [[ ! -f "tasks/oops_refnet/dflex_${env}.yaml" ]]; then
      echo "ERROR: no task file tasks/oops_refnet/dflex_${env}.yaml (valid: hopper ant humanoid snu_humanoid)" >&2
      exit 1
    fi

    echo "============================================================"
    echo "Launching OOPS (refnet) for env: ${env}"
    echo "============================================================"

    if [[ "$DRY_RUN" != "1" ]]; then
        wait_until_room_for_next_batch
    fi

    python spawner.py \
        --task_name oops_refnet \
        --docker \
        --docker_image /home/users/c/candidor/docker/mineral.sif \
        --deployment slurm \
        "${SPAWN_FLAGS[@]}" \
        --env_files "dflex_${env}.yaml"
done
