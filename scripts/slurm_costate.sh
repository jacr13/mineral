#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="run_costate_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

USER="candidor"
POLL_SECONDS=60
MAX_JOBS=100
BATCH_JOBS=""
SETTLE_SECONDS=5

ENVS=(
  "dflex_ant"
  "dflex_hopper"
  "dflex_humanoid"
  "dflex_snu_humanoid"
)

PARAMS_AGENT_BASE=(
  "SAPO"
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
  echo "Launching Costate env: ${env}"
  echo "MAX_JOBS=${MAX_JOBS}"
  echo "============================================================"

  git pull
  for base_algo in "${PARAMS_AGENT_BASE[@]}"; do
    if [[ -n "$BATCH_JOBS" ]]; then
      wait_until_room_for_next_batch "$MAX_JOBS" "$BATCH_JOBS"
    fi

    python spawner.py \
      --task_name costate \
      --docker \
      --docker_image /home/users/c/candidor/docker/mineral.sif \
      --deployment slurm \
      --runtime 6h \
      --no-cleanup \
      --sweep \
      --sweep_max 150 \
      --base_algo "${base_algo}" \
      --set "wandb.project=Costate_${base_algo}-${env}-slurm" \
      --set "agent.shac.max_agent_steps=100000000" \
      --env_files "${env}.yaml" \
      --deploy_now

    if [[ -z "$BATCH_JOBS" ]]; then
      sleep "$SETTLE_SECONDS"
      BATCH_JOBS="$(job_count)"
      echo "Detected batch jobs: ${BATCH_JOBS}"
    fi
  done
done
