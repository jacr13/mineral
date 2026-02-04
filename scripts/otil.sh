#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

USER="candidor"
POLL_SECONDS=60
MAX_JOBS=100
BATCH_JOBS=""          # empty = unknown until first submit
SETTLE_SECONDS=5       # give slurm time to show new jobs

# ENVS=(
#   "dflex_hopper"
#   "dflex_ant"
#   "dflex_humanoid"
#   "dflex_snu_humanoid"
# )

ENVS=(
  "ant_run"
  "soft_jumper"
  "hand_reorient"
)

PARAMS_CRITIC_RM=(
  "agent.otil.critic_reward_mapping=exp"
  "agent.otil.critic_reward_mapping=log_exp"
)

PARAMS_OT_COST_TYPE=(
  "agent.otil.loss_ot_cost_type=l2"
  "agent.otil.loss_ot_cost_type=cosine"
)

PARAMS_MLP_FEATURES_DIM=(
  "agent.otil.loss_mlp_features_dim=64"
  "agent.otil.loss_mlp_features_dim=null"
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

# Sanity: all param arrays must have same length
N="${#PARAMS_CRITIC_RM[@]}"
if [[ "${#PARAMS_OT_COST_TYPE[@]}" -ne "$N" ]]; then
  echo "ERROR: PARAMS_* arrays must have the same length." >&2
  echo "  PARAMS_CRITIC_RM=${#PARAMS_CRITIC_RM[@]}" >&2
  echo "  PARAMS_OT_COST_TYPE=${#PARAMS_OT_COST_TYPE[@]}" >&2
  exit 1
fi

for env in "${ENVS[@]}"; do
  echo "============================================================"
  echo "Launching env: ${env}"
  echo "WandB project: sweep-${env}-slurm-new"
  echo "MAX_JOBS=${MAX_JOBS}"
  echo "============================================================"

  git pull
  for base_algo in "${PARAMS_AGENT_BASE[@]}"; do
    echo "------------------------------------------------------------"
    echo "Base algorithm: ${base_algo}"
    echo "------------------------------------------------------------"
    for ((i=0; i<N; i++)); do
      critic_rm="${PARAMS_CRITIC_RM[$i]}"
      ot_cost="${PARAMS_OT_COST_TYPE[$i]}"
      mlp_feat="${PARAMS_MLP_FEATURES_DIM[$i]}"
      batch_tag="pair$((i+1))__${critic_rm##*=}__${ot_cost##*=}"

      echo "------------------------------------------------------------"
      echo "Config $((i+1))/${N}: ${batch_tag}"
      echo "  ${critic_rm}"
      echo "  ${ot_cost}"
      echo "------------------------------------------------------------"

      # If we already know how many jobs one batch creates, wait until there is room.
      if [[ -n "$BATCH_JOBS" ]]; then
        wait_until_room_for_next_batch "$MAX_JOBS" "$BATCH_JOBS"
      fi

      python spawner.py \
        --task_name otil \
        --docker \
        --docker_image /home/users/c/candidor/docker/mineral.sif \
        --deployment slurm \
        --runtime 6h \
        --no-cleanup \
        --sweep \
        --sweep_max 150 \
        --set "agent.name=OTIL/DFlexAnt${base_algo}" \
        --set "agent.shac.max_agent_steps=100000000" \
        --set "wandb.project=bestofk_OTIL_${base_algo}-${env}-slurm-new" \
        --set "${critic_rm}" \
        --set "${ot_cost}" \
        --set "${mlp_feat}" \
        --env_files "${env}.yaml" \
        --deploy_now

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

    if [[ -n "$BATCH_JOBS" ]]; then
      wait_until_room_for_next_batch "$MAX_JOBS" "$BATCH_JOBS"
    fi
    python spawner.py \
      --task_name otil \
      --docker \
      --docker_image /home/users/c/candidor/docker/mineral.sif \
      --deployment slurm \
      --runtime 6h \
      --no-cleanup \
      --sweep \
      --sweep_max 150 \
      --set "agent.name=OTIL/DFlexAnt${base_algo}" \
      --set "agent.shac.max_agent_steps=100000000" \
      --set "wandb.project=OTIL_${base_algo}-${env}-slurm-new" \
      --set "agent.otil.imitation_loss_type=l2" \
      --set "agent.otil.loss_mlp_features_dim=64" \
      --env_files "${env}.yaml" \
      --deploy_now

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
done
