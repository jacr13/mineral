#!/usr/bin/env bash
# Everything worth running on a free cluster, 12h per job.
#
# Each batch is independent and prints its own plan. Run the whole thing, or
# pick batches with BATCHES="A C".
#
#   bash scripts/rebuttal_overnight.sh
#   BATCHES="A" bash scripts/rebuttal_overnight.sh
#   DRY_RUN=1 bash scripts/rebuttal_overnight.sh      # show every command, submit nothing
#
# Totals at 12h each:
#   A  gradient path on the three envs it has not run on   36 jobs
#   B  SNU seed-stability, 12 fresh seeds                  12 jobs
#   D  rerun the ZO-PPO cells that never reached budget    24 jobs
#                                                    ---------------
#                                                          72 jobs
#
# Batch C (soft-DTW at 26M) was considered and dropped: the paper reports at
# 15M, its figures truncate at 15M, and the reported numbers were verified to
# match the 15M truncation -- so the existing comparison is already matched and
# a 26M run would answer a question the paper does not ask.
set -euo pipefail

export RUNTIME="${RUNTIME:-12h}"          # shared-gpu partition ceiling
export CKPT_EVERY="${CKPT_EVERY:-2000}"   # ckpt_every=100 unpruned is what filled the disk
export MAX_JOBS="${MAX_JOBS:-100}"
DRY_RUN="${DRY_RUN:-0}"
export DRY_RUN
BATCHES="${BATCHES:-A B D}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"

banner() { echo; echo "============================================================"; echo "$1"; echo "============================================================"; }

# --- A. transport-plan gradient on hopper / ant / humanoid -------------------
# SNU already ran and is the headline result (fixed coupling 0.70 vs
# through-plan 0.30, non-overlapping). Three more envs turn one data point
# into a pattern, which is what a reviewer will want before believing that
# the implementation's deviation from the paper's text is a *better* choice.
if [[ " $BATCHES " == *" A "* ]]; then
  banner "A. gradient path: hopper, ant, humanoid  (3 envs x 2 arms x 6 seeds = 36 jobs)"
  ENVS="dflex_hopper dflex_ant dflex_humanoid" bash scripts/otil_gradpath.sh
fi

# --- B. SNU seed stability ---------------------------------------------------
# 12 fresh seeds at the reported FOCUS-OT-L2 config, disjoint from the reported
# six, to characterize the bimodality found in the rebuttal runs (2/6 collapsed
# where the reported batch had 0/6).
if [[ " $BATCHES " == *" B "* ]]; then
  banner "B. SNU Humanoid seed stability  (12 fresh seeds = 12 jobs)"
  cmd=(python spawner.py
    --task_name otil --docker
    --docker_image "${DOCKER_IMAGE:-/home/users/c/candidor/docker/mineral.sif}"
    --deployment slurm --runtime "$RUNTIME"
    --gpu_constraint "${GPU_CONSTRAINT:-COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE}"
    --no-cleanup --sweep --sweep_max 150 --base_algo SAPO
    --set "wandb.project=seedstability_OTIL_SAPO-dflex_snu_humanoid-slurm"
    --set "agent.shac.max_agent_steps=15000000"
    --set "agent.ckpt_every=${CKPT_EVERY}"
    --set "agent.otil.imitation_loss_type=ot"
    --set "agent.otil.loss_ot_cost_type=l2"
    --set "agent.otil.loss_best_of_k_k=8"
    --set "agent.otil.loss_best_of_k_tau=0.5"
    --set "agent.otil.loss_mlp_features_dim=64"
    --set "agent.otil.loss_ot_detach_plan=true"
    --set "agent.otil.critic_reward_mapping=exp"
    --set "agent.otil.loss_use_huber_speedup=false"
    --set "agent.otil.loss_use_detached_prev_obs=false"
    --set "agent.otil.critic_disabled=false"
    --set "agent.otil.critic_reward_shapping=true"
    --env_files "dflex_snu_humanoid_seedstability.yaml"
    --deploy_now)
  if [[ "$DRY_RUN" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
fi

# --- D. ZO-PPO cells that never reached the stated budget --------------------
# plotter.py sets a method's final x to min(steps[-1]) over its seeds, so one
# seed that died early drags the whole method's reported endpoint down. Four
# cells are affected, and Table 13 reports them "at the final interaction
# budget" while they actually sit at 55-82% of it:
#
#   SNU      ZO-FOCUS-PPO-OT-L2    8.28M of 15M
#   SNU      ZO-FOCUS-PPO-OT-Cos   8.61M of 15M
#   SNU      ZO-FOCUS-PPO-L2      12.26M of 15M
#   Humanoid ZO-FOCUS-PPO-OT-Cos  10.50M of 15M
#
# This inflates the FOCUS-vs-zeroth-order gap in the paper's favour, so it is a
# correctness fix to an existing claim rather than a new experiment -- the
# highest-stakes batch here. The originals were capped at 7h30m; 12h should let
# them finish. Whole cells are rerun (6 seeds each) rather than cherry-picking
# the short seeds, so each cell is internally consistent.
if [[ " $BATCHES " == *" D "* ]]; then
  banner "D. rerun short ZO-PPO cells  (4 cells x 6 seeds = 24 jobs)"
  #        env                  variant       loss  cost    mlp
  ROWS_D=( "dflex_snu_humanoid:ppo_ot_l2:ot:l2:64"
           "dflex_snu_humanoid:ppo_ot_cos:ot:cosine:null"
           "dflex_snu_humanoid:ppo_l2:l2:l2:null"
           "dflex_humanoid:ppo_ot_cos:ot:cosine:null" )
  for row in "${ROWS_D[@]}"; do
    IFS=':' read -r env variant loss cost mlp <<< "$row"
    steps=15000000
    echo "  ${env} / ${variant}"
    cmd=(python spawner.py
      --task_name otilppo --docker
      --docker_image "${DOCKER_IMAGE:-/home/users/c/candidor/docker/mineral.sif}"
      --deployment slurm --runtime "$RUNTIME"
      --gpu_constraint "${GPU_CONSTRAINT:-COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE}"
      --no-cleanup --sweep --sweep_max 150
      --set "wandb.project=rerun_OTIL_PPO-${env}-slurm"
      --set "agent.ppo.max_agent_steps=${steps}"
      --set "agent.ckpt_every=${CKPT_EVERY}"
      --set "agent.otil.imitation_loss_type=${loss}"
      --set "agent.otil.loss_ot_cost_type=${cost}"
      --set "agent.otil.loss_mlp_features_dim=${mlp}"
      --set "agent.otil.critic_reward_mapping=exp"
      --set "agent.otil.critic_reward_shapping=false"
      --set "agent.otil.input_type=state_state"
      --set "agent.otil.loss_best_of_k_k=8"
      --set "agent.otil.loss_use_huber_speedup=false"
      --set "agent.otil.loss_use_detached_prev_obs=false"
      # *_rerun.yaml, not ${env}.yaml: the original carries
      # spawner.runtime=7h30m, which overrides --runtime and is what killed
      # these seeds early in the first place.
      --env_files "${env}_rerun.yaml"
      --deploy_now)
    if [[ "$DRY_RUN" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
  done
fi

echo
echo "Submitted. Projects to watch:"
echo "  gradpath_OTIL_SAPO-dflex_{hopper,ant,humanoid}-slurm"
echo "  seedstability_OTIL_SAPO-dflex_snu_humanoid-slurm"
echo "  rerun_OTIL_PPO-dflex_{snu_humanoid,humanoid}-slurm"
echo
echo "Then: python plotter/rebuttal_plots.py --refresh"
