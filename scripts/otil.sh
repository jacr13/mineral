#!/usr/bin/env bash
set -euo pipefail

USER="candidor"
POLL_SECONDS=60

ENVS=(
  "dflex_ant"
  "dflex_hopper"
  "dflex_humanoid"
  "dflex_snu_humanoid"
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
  "agent.otil.loss_mlp_features_dim=[32,64]"
  "agent.otil.loss_mlp_features_dim=null"
)

wait_for_jobs_to_finish() {
  while true; do
    count="$(squeue -u "$USER" | grep -wc "$USER" || true)"
    if [[ "$count" -eq 0 ]]; then
      echo "No jobs detected for $USER. Continuing."
      break
    fi
    echo "Jobs still running for $USER: $count. Sleeping ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
  done
}

# Sanity: all param arrays must have same length
N="${#PARAMS_CRITIC_RM[@]}"
if [[ "${#PARAMS_OT_COST_TYPE[@]}" -ne "$N" || "${#PARAMS_MLP_FEATURES_DIM[@]}" -ne "$N" ]]; then
  echo "ERROR: PARAMS_* arrays must have the same length." >&2
  echo "  PARAMS_CRITIC_RM=${#PARAMS_CRITIC_RM[@]}" >&2
  echo "  PARAMS_OT_COST_TYPE=${#PARAMS_OT_COST_TYPE[@]}" >&2
  echo "  PARAMS_MLP_FEATURES_DIM=${#PARAMS_MLP_FEATURES_DIM[@]}" >&2
  exit 1
fi

for env in "${ENVS[@]}"; do
  echo "============================================================"
  echo "Launching env: ${env}"
  echo "WandB project: sweep-${env}-slurm-new"
  echo "============================================================"

  git pull

  for ((i=0; i<N; i++)); do
    critic_rm="${PARAMS_CRITIC_RM[$i]}"
    ot_cost="${PARAMS_OT_COST_TYPE[$i]}"
    mlp_dim="${PARAMS_MLP_FEATURES_DIM[$i]}"

    # Optional readable tag for W&B name
    batch_tag="pair$((i+1))__${critic_rm##*=}__${ot_cost##*=}__mlp_$(echo "${mlp_dim##*=}" | tr -d '[] ,')"

    echo "------------------------------------------------------------"
    echo "Submitting paired config $((i+1))/${N}: ${batch_tag}"
    echo "  ${critic_rm}"
    echo "  ${ot_cost}"
    echo "  ${mlp_dim}"
    echo "------------------------------------------------------------"

    python spawner.py \
      --task_name otil \
      --docker \
      --docker_image /home/users/c/candidor/docker/mineral.sif \
      --deployment slurm \
      --runtime 6h \
      --no-cleanup \
      --sweep \
      --sweep_max 150 \
      --set "wandb.project=sweep-${env}-slurm-new" \
      --set "${critic_rm}" \
      --set "${ot_cost}" \
      --set "${mlp_dim}" \
      --env_files "${env}.yaml" \
      --deploy_now

    echo "Waiting for SLURM jobs to finish before next paired config..."
    wait_for_jobs_to_finish
  done
done
