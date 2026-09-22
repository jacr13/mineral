#!/usr/bin/env bash
# Ablation over the best-of-K pool size (agent.otil.loss_best_of_k_k) for the
# two OT-based FOCUS/OTIL variants (FOCUS-OT-L2 and FOCUS-OT-Cos). Mirrors
# scripts/otilsac.sh's launch pattern, restricted to a single environment and
# a K sweep instead of a variant sweep.
#
# Default footprint: 1 env x 2 variants x 6 K values x 3 seeds = 36 jobs.
# Trim K_VALUES / SEEDS below to shrink that further if time is tight.
set -euo pipefail

LOG_FILE="run_otil_k_ablation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

USER="candidor"
POLL_SECONDS=60
MAX_JOBS=300
BATCH_JOBS=""
SETTLE_SECONDS=5

# "sac" or "ppo". Both variants use best-of-K matching (imitation_loss_type=ot);
# SAC is the default here since it is the cheaper of the two to get an early
# read on K sensitivity.
BASE_ALGO="sac"

ENV="dflex_ant"
RUNTIME="8h"
MAX_AGENT_STEPS=10000000

SEEDS=(100 110 120)

# K values to sweep, bracketing the FOCUS default of K=8.
K_VALUES=(1 2 4 8 16 32)

# variant name|OT cost type|embedding dim|reward mapping
# (matches the focus_ot_l2 / focus_ot_cos rows of scripts/otilsac.sh, minus
# the plain-L2 non-OT variant, which the K ablation doesn't target)
VARIANTS=(
  "ot_l2|l2|64|exp"
  "ot_cos|cosine|null|log_exp"
)

case "$BASE_ALGO" in
  sac)
    TASK_NAME="otilsac"
    WANDB_PROJECT_PREFIX="OTIL_K_ABLATION_SAC"
    STEPS_KEY="agent.sac.max_agent_steps"
    LOGDIR_PREFIX="OTIL_K_ABLATION_SAC"
    ;;
  ppo)
    TASK_NAME="otilppo"
    WANDB_PROJECT_PREFIX="OTIL_K_ABLATION_PPO"
    STEPS_KEY="agent.ppo.max_agent_steps"
    LOGDIR_PREFIX="OTIL_K_ABLATION_PPO"
    ;;
  *)
    echo "ERROR: unknown BASE_ALGO '${BASE_ALGO}' (expected sac or ppo)." >&2
    exit 1
    ;;
esac

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

echo "============================================================"
echo "Launching OTIL K ablation (${BASE_ALGO}) on ${ENV}"
echo "K values: ${K_VALUES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "============================================================"

git pull

for variant_row in "${VARIANTS[@]}"; do
  IFS="|" read -r variant ot_cost mlp_dim reward_mapping <<<"${variant_row}"

  for k in "${K_VALUES[@]}"; do
    condition="${variant}_k${k}"

    echo "------------------------------------------------------------"
    echo "Config: ${condition}"
    echo "  ot_cost=${ot_cost}  embedding_dim=${mlp_dim}  reward_mapping=${reward_mapping}  K=${k}"
    echo "------------------------------------------------------------"

    for seed in "${SEEDS[@]}"; do
      echo "  Seed: ${seed}"
      if [[ -n "$BATCH_JOBS" ]]; then
        wait_until_room_for_next_batch "$MAX_JOBS" "$BATCH_JOBS"
      fi

      python spawner.py \
        --task_name "${TASK_NAME}" \
        --docker \
        --docker_image /home/users/c/candidor/docker/mineral.sif \
        --deployment slurm \
        --runtime "${RUNTIME}" \
        --no-cleanup \
        --set "seed=${seed}" \
        --set "logdir=workdir/${LOGDIR_PREFIX}_${ENV}_${RUN_STAMP}/${condition}/seed_${seed}" \
        --set "wandb.project=${WANDB_PROJECT_PREFIX}-${ENV}-slurm" \
        --set "${STEPS_KEY}=${MAX_AGENT_STEPS}" \
        --set "agent.otil.input_type=state_state" \
        --set "agent.otil.imitation_loss_type=ot" \
        --set "agent.otil.loss_ot_cost_type=${ot_cost}" \
        --set "agent.otil.loss_mlp_features_dim=${mlp_dim}" \
        --set "agent.otil.critic_reward_mapping=${reward_mapping}" \
        --set "agent.otil.critic_reward_shapping=false" \
        --set "agent.otil.loss_best_of_k_k=${k}" \
        --env_files "${ENV}.yaml" \
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

echo "============================================================"
echo "All jobs submitted. Once they finish, analyze with:"
echo "  python scripts/analyze_otil_k_ablation.py workdir/${LOGDIR_PREFIX}_${ENV}_${RUN_STAMP}"
echo "============================================================"
