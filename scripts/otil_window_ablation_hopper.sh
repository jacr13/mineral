#!/usr/bin/env bash
# Hopper-only follow-up to the best-of-K ablation: does shrinking the
# best-of-K matching window (relative to the gait period) make K matter more?
#
# agent.shac.horizon_len doubles as both the RL rollout length AND the
# best-of-K window T (mineral/agents/otil/otil.py: T=horizon_len when
# input_type=="state", which this sweep uses). The original bestofk sweep
# used horizon_len=32; Hopper's gait period is ~37 steps (docs/phase_rescue),
# so that window already covers ~87% of a full cycle -- there's little phase
# ambiguity left for more candidates (higher K) to resolve, which matches the
# muted K effect actually observed for Hopper. This sweep drops horizon_len
# to 16 (~43% of the period, genuinely sub-cycle) to see if K's effect on
# final return grows once there's real ambiguity to resolve.
#
# Confound, not swept away: horizon_len is also the RL rollout/truncation
# length, so this changes more than just the matching window -- a smaller
# horizon_len itself has known effects on SHAC-style differentiable-sim
# training (bias/variance of the truncated return estimator) independent of
# best-of-K. Any K-effect difference vs. the horizon_len=32 sweep should be
# read with that in mind, not attributed to window size alone.
#
# Separate wandb project from the main bestofk_OTIL_SAPO sweep (different
# horizon_len = a different experiment, not more seeds of the same one) --
# don't merge these into plotter/bestofk_ablation.py's fetch_data.
#
# Footprint: 1 env x 5 K values x 2 costs x 6 seeds = 60 jobs.
set -euo pipefail

LOG_FILE="run_otil_window_ablation_hopper_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

USER="candidor"
POLL_SECONDS=60
MAX_JOBS=300
BATCH_JOBS=""
SETTLE_SECONDS=5

ENV="hopper"
HORIZON_LEN=16
RUNTIME="3h30m"
MAX_AGENT_STEPS=5000000
WANDB_PROJECT="bestofk_window16_OTIL_SAPO-dflex_hopper-slurm-new"

SEEDS=(100 110 120 1000 1100 1200)
K_VALUES=(1 2 4 8 16)
COSTS=(l2 cosine)

declare -A MLP_DIM=( [l2]=64 [cosine]=null )
declare -A REWARD_MAPPING=( [l2]=exp [cosine]=log_exp )

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
echo "Hopper window ablation: horizon_len=${HORIZON_LEN} (vs. original 32)"
echo "K values: ${K_VALUES[*]}  Costs: ${COSTS[*]}  Seeds: ${SEEDS[*]}"
echo "wandb project: ${WANDB_PROJECT}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "============================================================"

git pull

for cost in "${COSTS[@]}"; do
  mlp_dim="${MLP_DIM[$cost]}"
  reward_mapping="${REWARD_MAPPING[$cost]}"

  for k in "${K_VALUES[@]}"; do
    condition="k${k}_${cost}"

    echo "------------------------------------------------------------"
    echo "Config: ${condition}  horizon_len=${HORIZON_LEN}"
    echo "------------------------------------------------------------"

    for seed in "${SEEDS[@]}"; do
      echo "  Seed: ${seed}"
      if [[ -n "$BATCH_JOBS" ]]; then
        wait_until_room_for_next_batch "$MAX_JOBS" "$BATCH_JOBS"
      fi

      python spawner.py \
        --task_name otil \
        --base_algo SAPO \
        --docker \
        --docker_image /home/users/c/candidor/docker/mineral.sif \
        --deployment slurm \
        --runtime "${RUNTIME}" \
        --no-cleanup \
        --set "seed=${seed}" \
        --set "logdir=workdir/bestofk_window16_${ENV}_${RUN_STAMP}/${condition}/seed_${seed}" \
        --set "wandb.project=${WANDB_PROJECT}" \
        --set "agent.shac.max_agent_steps=${MAX_AGENT_STEPS}" \
        --set "agent.shac.horizon_len=${HORIZON_LEN}" \
        --set "agent.otil.imitation_loss_type=ot" \
        --set "agent.otil.loss_ot_cost_type=${cost}" \
        --set "agent.otil.loss_best_of_k_k=${k}" \
        --set "agent.otil.loss_mlp_features_dim=${mlp_dim}" \
        --set "agent.otil.critic_reward_mapping=${reward_mapping}" \
        --set "agent.otil.loss_use_huber_speedup=false" \
        --set "agent.otil.loss_use_detached_prev_obs=false" \
        --set "agent.otil.actor_value_return_type=min" \
        --set "agent.otil.critic_reward_normalize=false" \
        --env_files "dflex_${ENV}.yaml" \
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
echo "All jobs submitted. Once they finish, analyze against the horizon_len=32"
echo "Hopper results already in plotter/bestofk_ablation/ (project ${WANDB_PROJECT})."
echo "============================================================"
