#!/usr/bin/env bash
# Extend the existing best-of-K ablation (wandb projects
# jacr/bestofk_OTIL_SAPO-dflex_{env}-slurm-new) with 3 additional seeds.
#
# The original sweep (OTIL/FOCUS-SAPO base, K in {1,2,4,8,16}, OT cost in
# {l2, cosine}) only used seeds 100/110/120 -- see
# scripts/rerun_missing_bestofk_seeds.sh for the one gap in that original
# run. This script adds seeds 1000/1100/1200 (matching the 6-seed convention
# used elsewhere, e.g. scripts/otilsac.sh) on top of it, submitting straight
# into the same wandb projects so plotter/bestofk_ablation.py picks them up
# automatically and the per-condition seed count goes from 3 to 6 (tighter
# bootstrap CIs).
#
# Footprint: 4 envs x 5 K values x 2 costs x 3 seeds = 120 jobs. Trim
# ENVS/K_VALUES/SEEDS below if that's too much for the time you have.
#
# IMPORTANT: --task_name otil alone resolves agent=OTIL/DFlexAntSHAC (a plain,
# untuned Critic/ELU/Adam setup), NOT the EnsembleCritic/SiLU/AdamW/autoent
# config the original "bestofk_OTIL_SAPO" sweep actually used
# (agent=OTIL/DFlexAntSAPO). Confirmed via direct Hydra compose: the two
# resolve to genuinely different networks/optimizers despite both nominally
# extending the same SHAC/DFlexAnt base. An earlier version of this script
# omitted --base_algo SAPO and every one of its 120 jobs silently trained
# with the wrong, less stable config -- ~70% diverged. Do not remove
# --base_algo SAPO below.
set -euo pipefail

LOG_FILE="run_otil_k_ablation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

USER="candidor"
POLL_SECONDS=60
MAX_JOBS=300
BATCH_JOBS=""
SETTLE_SECONDS=5

SEEDS=(1000 1100 1200)

# K values matching the original sweep.
K_VALUES=(1 2 4 8 16)

# ot_cost -> embedding dim|reward mapping, matching the original sweep
# (verified against the surviving runs' resolved wandb configs).
declare -A MLP_DIM=( [l2]=64 [cosine]=null )
declare -A REWARD_MAPPING=( [l2]=exp [cosine]=log_exp )
COSTS=(l2 cosine)

# env -> slurm runtime, matching the original sweep's per-env wall-clock cap.
declare -A RUNTIME=( [hopper]="3h30m" [ant]="4h" [humanoid]="7h30m" [snu_humanoid]="7h30m" )
ENVS=(hopper ant humanoid snu_humanoid)

# env -> max_agent_steps: the original sweep set this to 100M (never hit --
# every run was actually stopped by the wall-clock runtime above). Set from
# eyeballing the training curves in plotter/bestofk_ablation/ -- performance
# is already flat well before the original 11-38M step range, so these are
# picked as "clearly converged" points, not just "matches what old runs
# reached": hopper 5M, ant 10M, humanoid 15M, snu_humanoid 20M.
declare -A MAX_AGENT_STEPS=( [hopper]=5000000 [ant]=10000000 [humanoid]=15000000 [snu_humanoid]=20000000 )

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
echo "Extending bestofk OTIL/FOCUS-SAPO ablation with seeds: ${SEEDS[*]}"
echo "Envs: ${ENVS[*]}  K values: ${K_VALUES[*]}  Costs: ${COSTS[*]}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "============================================================"

git pull

for env in "${ENVS[@]}"; do
  runtime="${RUNTIME[$env]}"
  max_agent_steps="${MAX_AGENT_STEPS[$env]}"

  for cost in "${COSTS[@]}"; do
    mlp_dim="${MLP_DIM[$cost]}"
    reward_mapping="${REWARD_MAPPING[$cost]}"

    for k in "${K_VALUES[@]}"; do
      condition="k${k}_${cost}"

      echo "------------------------------------------------------------"
      echo "Config: env=${env} K=${k} ot_cost=${cost} runtime=${runtime} max_agent_steps=${max_agent_steps}"
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
          --runtime "${runtime}" \
          --no-cleanup \
          --set "seed=${seed}" \
          --set "logdir=workdir/bestofk_extra_seeds_${RUN_STAMP}/${env}/${condition}/seed_${seed}" \
          --set "wandb.project=bestofk_OTIL_SAPO-dflex_${env}-slurm-new" \
          --set "agent.shac.max_agent_steps=${max_agent_steps}" \
          --set "agent.otil.imitation_loss_type=ot" \
          --set "agent.otil.loss_ot_cost_type=${cost}" \
          --set "agent.otil.loss_best_of_k_k=${k}" \
          --set "agent.otil.loss_mlp_features_dim=${mlp_dim}" \
          --set "agent.otil.critic_reward_mapping=${reward_mapping}" \
          --set "agent.otil.loss_use_huber_speedup=false" \
          --set "agent.otil.loss_use_detached_prev_obs=false" \
          --set "agent.otil.actor_value_return_type=min" \
          --set "agent.otil.critic_reward_normalize=false" \
          --env_files "dflex_${env}.yaml" \
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
done

echo "============================================================"
echo "All jobs submitted. Once they finish, refresh the plots/tables with:"
echo "  cd plotter && python bestofk_ablation.py"
echo "============================================================"
