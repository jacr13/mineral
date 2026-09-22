#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

USER="candidor"
POLL_SECONDS=60
MAX_JOBS=100
BATCH_JOBS=""          # empty = unknown until first submit
SETTLE_SECONDS=5       # give slurm time to show new jobs

ENVS=(
  "dflex_hopper"
  "dflex_ant"
  "dflex_humanoid"
  "dflex_snu_humanoid"
)

# Launch only OOPS via its task config directory.
BASELINE_NAMES=("oops")
BASELINE_TASK_NAMES=("oops")
BASELINE_STEPS_KEYS=("agent.ddpg.max_agent_steps")
BASELINE_EXTRA_SETS=("")

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

N_BASELINES="${#BASELINE_NAMES[@]}"

for ((b=0; b<N_BASELINES; b++)); do
  name="${BASELINE_NAMES[$b]}"
  task_name="${BASELINE_TASK_NAMES[$b]}"
  steps_key="${BASELINE_STEPS_KEYS[$b]}"
  extra_set="${BASELINE_EXTRA_SETS[$b]}"

  for env in "${ENVS[@]}"; do
    echo "============================================================"
    echo "Launching baseline: ${name}  (task_name=${task_name})"
    echo "Env: ${env}"
    echo "WandB project: ${name}-sweep-${env}-slurm"
    echo "MAX_JOBS=${MAX_JOBS}"
    echo "============================================================"

    git pull

    # If we already know how many jobs one batch creates, wait until there is room.
    if [[ -n "$BATCH_JOBS" ]]; then
        wait_until_room_for_next_batch "$MAX_JOBS" "$BATCH_JOBS"
    fi

    # Snapshot the queue right before submitting, so the batch size below is a
    # delta rather than an absolute count -- correct even if other jobs (e.g.
    # an opolo run launched earlier, outside this script) are already queued.
    jobs_before="$(job_count)"

    set_args=(
        --set "wandb.project=${name}-sweep-${env}-slurm"
        --set "${steps_key}=100000000"
    )
    if [[ -n "$extra_set" ]]; then
        set_args+=(--set "$extra_set")
    fi

    python spawner.py \
        --task_name "${task_name}" \
        --docker \
        --docker_image /home/users/c/candidor/docker/mineral.sif \
        --deployment slurm \
        --runtime 6h \
        --no-cleanup \
        --sweep \
        --sweep_max 150 \
        "${set_args[@]}" \
        --env_files "${env}.yaml" \
        --deploy_now

    # Learn batch size from the first ever submission, as the delta in queue
    # size (not its raw post-submission value), so pre-existing jobs from
    # other launches don't get miscounted as part of this batch.
    if [[ -z "$BATCH_JOBS" ]]; then
        sleep "$SETTLE_SECONDS"
        jobs_after="$(job_count)"
        BATCH_JOBS=$(( jobs_after - jobs_before ))
        echo "Detected batch jobs (delta): ${BATCH_JOBS} (before=${jobs_before}, after=${jobs_after})"

        if [[ "$BATCH_JOBS" -le 0 ]]; then
        echo "WARNING: Detected non-positive batch size (${BATCH_JOBS}). SLURM may be delayed, or submission failed." >&2
        fi
        if [[ "$BATCH_JOBS" -gt "$MAX_JOBS" ]]; then
        echo "WARNING: batch_jobs=${BATCH_JOBS} > MAX_JOBS=${MAX_JOBS}. The cap cannot be enforced with this MAX_JOBS." >&2
        fi
    fi
  done
done
