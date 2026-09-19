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

# Per-env step budget (matches plotter/plotter.py's MAX_STEPS_PER_ENV) and the
# wall-clock budget sized to reliably reach it for the slowest observed config
# in that env (throughput varies a lot within the same env -- not cleanly by
# critic_disabled, likely dominated by per-job cluster/GPU contention). Derived
# from measured steps/sec on the prior sweep, which used max_runtime=1h/2h/7h30m
# and left most runs far short of target (e.g. Hopper only reached ~1.5-3M of 10M).
# Both arrays must stay index-aligned with ENVS above.
#
# Humanoid and SNU Humanoid are capped at 12h (the shared-gpu partition
# ceiling in spawner.py's CALIBERS) rather than the ~17.5h Humanoid's slowest
# observed config would actually need for 15M, to avoid routing to the
# private-cui-gpu / private-kalousis-gpu partition that no other task config
# in this repo uses. At 12h most seeds clear 15M comfortably (median
# throughput implies ~30M+), but the single worst-throughput outlier may still
# fall a bit short -- accepted tradeoff, confirmed with the user.
#
# These values are also set directly in each tasks/otil/dflex_*.yaml's
# `spawner.runtime`, which takes precedence over this script's --runtime flag
# (spawner.py: a config's own spawner.runtime always overrides the CLI value)
# -- kept in sync here so the two don't silently disagree.
MAX_STEPS=(
  10000000  # dflex_hopper
  10000000  # dflex_ant
  15000000  # dflex_humanoid
  15000000  # dflex_snu_humanoid
)
RUNTIMES=(
  "8h"   # dflex_hopper
  "8h"   # dflex_ant
  "12h"  # dflex_humanoid
  "12h"  # dflex_snu_humanoid
)

# ENVS=(
#   # "rewarped_ant_run"
#   "rewarped_soft_jumper"
#   # "rewarped_hand_reorient"
# )

# Full critic vs no-critic ablation matrix (critic_reward_shapping=true
# throughout). Index-aligned across all five PARAMS_* arrays below; each row's
# method name (as critic_no_critic.py's METHOD_ORDER names it) is noted inline.
# Previously only rows 0-3 were active (all critic_disabled=true) plus a
# shaping=false variant of exp/logexp in the old commented block -- the
# critic-enabled, shaping=true rows for exp/logexp x l2/cosine (now rows 0-3)
# were never actually launched.
PARAMS_CRITIC_RM=(
  "agent.otil.critic_reward_mapping=exp"      # critic-exp-l2
  "agent.otil.critic_reward_mapping=exp"      # critic-exp-cos
  "agent.otil.critic_reward_mapping=log_exp"  # critic-logexp-l2
  "agent.otil.critic_reward_mapping=log_exp"  # critic-logexp-cos
  "agent.otil.critic_reward_mapping=neg"      # critic-neg-l2
  "agent.otil.critic_reward_mapping=exp"      # no-critic-exp-l2
  "agent.otil.critic_reward_mapping=exp"      # no-critic-exp-cos
  "agent.otil.critic_reward_mapping=log_exp"  # no-critic-logexp-l2
  "agent.otil.critic_reward_mapping=log_exp"  # no-critic-logexp-cos
  "agent.otil.critic_reward_mapping=neg"      # no-critic-neg-l2
)

PARAMS_OT_COST_TYPE=(
  "agent.otil.loss_ot_cost_type=l2"
  "agent.otil.loss_ot_cost_type=cosine"
  "agent.otil.loss_ot_cost_type=l2"
  "agent.otil.loss_ot_cost_type=cosine"
  "agent.otil.loss_ot_cost_type=l2"
  "agent.otil.loss_ot_cost_type=l2"
  "agent.otil.loss_ot_cost_type=cosine"
  "agent.otil.loss_ot_cost_type=l2"
  "agent.otil.loss_ot_cost_type=cosine"
  "agent.otil.loss_ot_cost_type=l2"
)

PARAMS_MLP_FEATURES_DIM=(
  "agent.otil.loss_mlp_features_dim=64"
  "agent.otil.loss_mlp_features_dim=null"
  "agent.otil.loss_mlp_features_dim=64"
  "agent.otil.loss_mlp_features_dim=null"
  "agent.otil.loss_mlp_features_dim=64"
  "agent.otil.loss_mlp_features_dim=64"
  "agent.otil.loss_mlp_features_dim=null"
  "agent.otil.loss_mlp_features_dim=64"
  "agent.otil.loss_mlp_features_dim=null"
  "agent.otil.loss_mlp_features_dim=64"
)

PARAMS_CRITIC_DISABLED=(
  "agent.otil.critic_disabled=false"
  "agent.otil.critic_disabled=false"
  "agent.otil.critic_disabled=false"
  "agent.otil.critic_disabled=false"
  "agent.otil.critic_disabled=false"
  "agent.otil.critic_disabled=true"
  "agent.otil.critic_disabled=true"
  "agent.otil.critic_disabled=true"
  "agent.otil.critic_disabled=true"
  "agent.otil.critic_disabled=true"
)

PARAMS_CRITIC_REWARD_SHAPPING=(
  "agent.otil.critic_reward_shapping=true"
  "agent.otil.critic_reward_shapping=true"
  "agent.otil.critic_reward_shapping=true"
  "agent.otil.critic_reward_shapping=true"
  "agent.otil.critic_reward_shapping=true"
  "agent.otil.critic_reward_shapping=true"
  "agent.otil.critic_reward_shapping=true"
  "agent.otil.critic_reward_shapping=true"
  "agent.otil.critic_reward_shapping=true"
  "agent.otil.critic_reward_shapping=true"
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

for env_idx in "${!ENVS[@]}"; do
  env="${ENVS[$env_idx]}"
  runtime="${RUNTIMES[$env_idx]}"
  max_steps="${MAX_STEPS[$env_idx]}"
  echo "============================================================"
  echo "Launching env: ${env}"
  echo "WandB project: sweep-${env}-slurm-new"
  echo "MAX_JOBS=${MAX_JOBS}"
  echo "runtime=${runtime}  max_agent_steps=${max_steps}"
  echo "============================================================"

  git pull
  for base_algo in "${PARAMS_AGENT_BASE[@]}"; do
    echo "------------------------------------------------------------"
    echo "Base algorithm: ${base_algo}"
    echo "------------------------------------------------------------"
    for ((i=0; i<N; i++)); do
      critic_rmapping="${PARAMS_CRITIC_RM[$i]}"
      ot_cost="${PARAMS_OT_COST_TYPE[$i]}"
      mlp_feat="${PARAMS_MLP_FEATURES_DIM[$i]}"
      critic_disabled="${PARAMS_CRITIC_DISABLED[$i]}"
      critic_rshapping="${PARAMS_CRITIC_REWARD_SHAPPING[$i]}"
      batch_tag="pair$((i+1))__${critic_rmapping##*=}__${ot_cost##*=}"

      echo "------------------------------------------------------------"
      echo "Config $((i+1))/${N}: ${batch_tag}"
      echo "  ${critic_rmapping}"
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
        --runtime "${runtime}" \
        --no-cleanup \
        --sweep \
        --sweep_max 150 \
        --base_algo "${base_algo}" \
        --set "wandb.project=new_v2_critic_OTIL_${base_algo}-${env}-slurm" \
        --set "agent.shac.max_agent_steps=${max_steps}" \
        --set "${critic_rmapping}" \
        --set "${ot_cost}" \
        --set "${mlp_feat}" \
        --set "${critic_disabled}" \
        --set "${critic_rshapping}" \
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

    # if [[ -n "$BATCH_JOBS" ]]; then
    #   wait_until_room_for_next_batch "$MAX_JOBS" "$BATCH_JOBS"
    # fi

    # if [[ -z "$BATCH_JOBS" ]]; then
    #   sleep "$SETTLE_SECONDS"
    #   BATCH_JOBS="$(job_count)"
    #   echo "Detected batch jobs (from empty queue): ${BATCH_JOBS}"

    #   if [[ "$BATCH_JOBS" -eq 0 ]]; then
    #     echo "WARNING: Detected 0 jobs after submission. SLURM may be delayed, or submission failed." >&2
    #   fi
    #   if [[ "$BATCH_JOBS" -gt "$MAX_JOBS" ]]; then
    #     echo "WARNING: batch_jobs=${BATCH_JOBS} > MAX_JOBS=${MAX_JOBS}. The cap cannot be enforced with this MAX_JOBS." >&2
    #   fi
    # fi
  done
done
