#!/usr/bin/env bash
# Resubmit the runs missing from the already-finished best-of-K ablation
# (wandb projects jacr/bestofk_OTIL_SAPO-dflex_{env}-slurm-new).
#
# Found via a full expected-vs-actual sweep over env x K x ot_cost x seed
# (5 K values x {l2,cosine} x 3 seeds x 4 envs = 120 runs expected):
# only one gap, hopper/K=8/cosine/seed=100. That run did finish and train
# normally (train_scores/episode_rewards ~4736, in line with its siblings)
# but never logged eval_scores/episode_rewards, so plotter/bestofk_ablation.py
# (which reads the eval metric) silently dropped it. Rerunning fresh rather
# than digging into why eval logging didn't fire for that one job.
#
# IMPORTANT: --task_name otil alone resolves agent=OTIL/DFlexAntSHAC (a plain,
# untuned Critic/ELU/Adam setup), NOT the EnsembleCritic/SiLU/AdamW/autoent
# config the original "bestofk_OTIL_SAPO" sweep actually used
# (agent=OTIL/DFlexAntSAPO) -- confirmed via direct Hydra compose. An earlier
# version of this script omitted --base_algo SAPO; its rerun of this exact
# seed came back at 3655 vs. its siblings' ~4750, silently using the wrong
# config. Do not remove --base_algo SAPO below.
#
# Re-run "cd plotter && python bestofk_ablation.py" once this finishes.
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

# env -> max_agent_steps, picked from eyeballing the training curves in
# plotter/bestofk_ablation/ -- performance is already flat well before this
# point (see scripts/otil_k_ablation.sh for the same values/rationale).
declare -A MAX_AGENT_STEPS=( [hopper]=5000000 [ant]=10000000 [humanoid]=15000000 [snu_humanoid]=20000000 )

# env -> slurm runtime, matching the original sweep's per-env wall-clock cap.
declare -A RUNTIME=( [hopper]="3h30m" [ant]="4h" [humanoid]="7h30m" [snu_humanoid]="7h30m" )

for row in "${MISSING[@]}"; do
  IFS="|" read -r env k ot_cost seed <<<"${row}"
  mlp_dim="${MLP_DIM[$ot_cost]}"
  reward_mapping="${REWARD_MAPPING[$ot_cost]}"
  max_agent_steps="${MAX_AGENT_STEPS[$env]}"
  runtime="${RUNTIME[$env]}"

  echo "------------------------------------------------------------"
  echo "Rerunning: env=${env} K=${k} ot_cost=${ot_cost} seed=${seed}"
  echo "------------------------------------------------------------"

  python spawner.py \
    --task_name otil \
    --base_algo SAPO \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm \
    --runtime "${runtime}" \
    --no-cleanup \
    --set "seed=${seed}" \
    --set "logdir=workdir/bestofk_rerun_${RUN_STAMP}/${env}_k${k}_${ot_cost}/seed_${seed}" \
    --set "wandb.project=bestofk_OTIL_SAPO-dflex_${env}-slurm-new" \
    --set "agent.shac.max_agent_steps=${max_agent_steps}" \
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

echo "Done. Once the job finishes, rerun: cd plotter && python bestofk_ablation.py"
