#!/usr/bin/env bash
# Resubmit the runs missing from the already-finished best-of-K ablation
# (wandb projects jacr/bestofk_OTIL_SAPO-dflex_{env}-slurm-new).
#
# Found via a full expected-vs-actual sweep over env x K x ot_cost x seed
# (5 K values x {l2,cosine} x 3 seeds x 4 envs = 120 runs expected):
# only one gap, hopper/K=8/cosine/seed=100. That run did finish and train
# normally (train_scores/episode_rewards ~4736, in line with its siblings)
# but never logged eval_scores/episode_rewards, so scripts/plot_bestofk_ablation.py
# (which reads the eval metric) silently dropped it. Rerunning fresh rather
# than digging into why eval logging didn't fire for that one job.
#
# Re-run scripts/plot_bestofk_ablation.py once this finishes to pick it up.
set -euo pipefail

LOG_FILE="run_bestofk_rerun_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

# env|K|ot_cost|seed -- add more rows here if a later check finds further gaps.
MISSING=(
  "hopper|8|cosine|100"
)

# ot_cost -> embedding dim|reward mapping, matching the original sweep
# (see resolved wandb configs of the surviving sibling runs).
declare -A MLP_DIM=( [l2]=64 [cosine]=null )
declare -A REWARD_MAPPING=( [l2]=exp [cosine]=log_exp )

for row in "${MISSING[@]}"; do
  IFS="|" read -r env k ot_cost seed <<<"${row}"
  mlp_dim="${MLP_DIM[$ot_cost]}"
  reward_mapping="${REWARD_MAPPING[$ot_cost]}"

  echo "------------------------------------------------------------"
  echo "Rerunning: env=${env} K=${k} ot_cost=${ot_cost} seed=${seed}"
  echo "------------------------------------------------------------"

  python spawner.py \
    --task_name otil \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm \
    --runtime "3h30m" \
    --no-cleanup \
    --set "seed=${seed}" \
    --set "logdir=workdir/bestofk_rerun_${RUN_STAMP}/${env}_k${k}_${ot_cost}/seed_${seed}" \
    --set "wandb.project=bestofk_OTIL_SAPO-dflex_${env}-slurm-new" \
    --set "agent.shac.max_agent_steps=100000000" \
    --set "agent.otil.imitation_loss_type=ot" \
    --set "agent.otil.loss_ot_cost_type=${ot_cost}" \
    --set "agent.otil.loss_best_of_k_k=${k}" \
    --set "agent.otil.loss_mlp_features_dim=${mlp_dim}" \
    --set "agent.otil.critic_reward_mapping=${reward_mapping}" \
    --set "agent.otil.loss_use_huber_speedup=false" \
    --set "agent.otil.loss_use_detached_prev_obs=false" \
    --set "agent.otil.actor_value_return_type=min" \
    --set "agent.otil.critic_reward_normalize=false" \
    --env_files "dflex_${env}.yaml" \
    --deploy_now
done

echo "Done. Once the job finishes, rerun: python scripts/plot_bestofk_ablation.py"
