#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

USER="candidor"
POLL_SECONDS=60
MAX_JOBS=75
BATCH_JOBS=""          # empty = unknown until first submit
SETTLE_SECONDS=5       # give slurm time to show new jobs

ENVS=(
  "dflex_ant"
  "dflex_hopper"
  "dflex_humanoid"
  "dflex_snu_humanoid"
)

job_count() {
  squeue -h -u "$USER" | wc -l | tr -d ' '
}

wait_until_room_for_next_batch() {
  local max_jobs="$1"
  local batch_jobs="$2"

  local threshold=$(( max_jobs - batch_jobs ))
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

for env in "${ENVS[@]}"; do
    echo "============================================================"
    echo "Launching env: ${env}"
    echo "WandB project: sweep-${env}-slurm-new"
    echo "MAX_JOBS=${MAX_JOBS}"
    echo "============================================================"

    git pull

    # If we already know how many jobs one batch creates, wait until there is room.
    if [[ -n "$BATCH_JOBS" ]]; then
        wait_until_room_for_next_batch "$MAX_JOBS" "$BATCH_JOBS"
    fi

    python spawner.py \
        --task_name ild \
        --docker \
        --docker_image /home/users/c/candidor/docker/mineral.sif \
        --deployment slurm \
        --runtime 6h \
        --no-cleanup \
        --sweep \
        --sweep_max 150 \
        --set "wandb.project=ild-sweep-${env}-slurm" \
        --set "agent.shac.max_agent_steps=100000000" \
        --set "agent.shac.max_epochs=50000" \
        --env_files "${env}.yaml" \
        --no-deploy_now

    # Learn batch size from the first ever submission (since you start from 0 jobs).
    if [[ -z "$BATCH_JOBS" ]]; then
        sleep "$SETTLE_SECONDS"
        BATCH_JOBS="$(job_count)"
        echo "Detected batch jobs (from empty queue): ${BATCH_JOBS}"

        if [[ "$BATCH_JOBS" -eq 0 ]]; then
        echo "WARNING: Detected 0 jobs after submission. SLURM may be delayed, or submission failed." >&2
        fi
        if [[ "$BATCH_JOBS" -gt "$MAX_JOBS" ]]; then
        echo "WARNING: batch_jobs=${BATCH_JOBS} > MAX_JOBS=${MAX_JOBS}. The cap cannot be enforced with this MAX_JOBS." >&2
        fi
    fi
done
