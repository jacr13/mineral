#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="run_otilppo_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

USER="candidor"
POLL_SECONDS=60
MAX_JOBS=100
BATCH_JOBS=""
SETTLE_SECONDS=5

ENVS=(
  "dflex_hopper"
  "dflex_ant"
  "dflex_humanoid"
  "dflex_snu_humanoid"
)

SEEDS=(100 110 120 1000 1100 1200)

# name|imitation loss|OT cost|embedding dim|reward mapping|extra cost shaping
COMBINATIONS=(
  "focus_l2|l2|l2|null|exp|false"
  "focus_ot_l2|ot|l2|64|exp|false"
  "focus_ot_cos|ot|cosine|null|log_exp|false"
)

job_count() {
  squeue -h -u "$USER" | wc -l | tr -d ' '
}

wait_until_room_for_next_batch() {
  local max_jobs="$1"
  local batch_jobs="$2"
  local threshold=$((max_jobs - batch_jobs))
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
  echo "Launching OTIL-PPO env: ${env}"
  echo "MAX_JOBS=${MAX_JOBS}"
  echo "============================================================"

  git pull
  for i in "${!COMBINATIONS[@]}"; do
    IFS="|" read -r condition imitation_loss ot_cost mlp_dim reward_mapping extra_cost_shaping <<<"${COMBINATIONS[$i]}"

    echo "------------------------------------------------------------"
    echo "Config $((i + 1))/${#COMBINATIONS[@]}: ${condition}"
    echo "  imitation_loss=${imitation_loss}"
    echo "  ot_cost=${ot_cost}"
    echo "  embedding_dim=${mlp_dim}"
    echo "  reward_mapping=${reward_mapping}"
    echo "------------------------------------------------------------"

    for seed in "${SEEDS[@]}"; do
      echo "  Seed: ${seed}"
      if [[ -n "$BATCH_JOBS" ]]; then
        wait_until_room_for_next_batch "$MAX_JOBS" "$BATCH_JOBS"
      fi

      python spawner.py \
        --task_name otilppo \
        --docker \
        --docker_image /home/users/c/candidor/docker/mineral.sif \
        --deployment slurm \
        --runtime 6h \
        --no-cleanup \
        --set "seed=${seed}" \
        --set "logdir=workdir/OTIL_PPO_${env}_${RUN_STAMP}/${condition}/seed_${seed}" \
        --set "wandb.project=OTIL_PPO-${env}-slurm" \
        --set "agent.ppo.max_agent_steps=10000000" \
        --set "agent.otil.input_type=state_state" \
        --set "agent.otil.imitation_loss_type=${imitation_loss}" \
        --set "agent.otil.loss_ot_cost_type=${ot_cost}" \
        --set "agent.otil.loss_mlp_features_dim=${mlp_dim}" \
        --set "agent.otil.critic_reward_mapping=${reward_mapping}" \
        --set "agent.otil.critic_reward_shapping=${extra_cost_shaping}" \
        --env_files "${env}.yaml" \
        --deploy_now

      if [[ -z "$BATCH_JOBS" ]]; then
        sleep "$SETTLE_SECONDS"
        BATCH_JOBS="$(job_count)"
        echo "Detected batch jobs (from empty queue): ${BATCH_JOBS}"
        if [[ "$BATCH_JOBS" -eq 0 ]]; then
          echo "WARNING: Detected 0 jobs after submission." >&2
        fi
        if [[ "$BATCH_JOBS" -gt "$MAX_JOBS" ]]; then
          echo "WARNING: batch_jobs=${BATCH_JOBS} > MAX_JOBS=${MAX_JOBS}." >&2
        fi
      fi
    done
  done
done
