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
#   F  simulator-derivative isolation (opt-in)             24 jobs
#   U  unbalanced OT, relaxed marginals (opt-in)           24 jobs
#                                                    ---------------
#                                             72 jobs, or 96 with F
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
BATCHES="${BATCHES:-A B D}"   # add F for the gradient-estimator isolation
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"

banner() { echo; echo "============================================================"; echo "$1"; echo "============================================================"; }

# Per-env interaction budget. The sub-scripts declare their own copy; batch B/D/F
# submit spawner directly from here and need it too.
declare -A MAX_STEPS=(
  [dflex_hopper]=10000000 [dflex_ant]=10000000
  [dflex_humanoid]=15000000 [dflex_snu_humanoid]=15000000
)
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
BATCH_JOBS="${BATCH_JOBS:-6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
USER_NAME="${USER_NAME:-candidor}"

job_count() { squeue -h -u "$USER_NAME" | wc -l | tr -d ' '; }
wait_for_room() {
  local need="$1"; local threshold=$(( MAX_JOBS - need ))
  (( threshold < 0 )) && threshold=0
  while true; do
    local n; n="$(job_count)"
    [[ "$n" -le "$threshold" ]] && { echo "Queue ${n} <= ${threshold}; room for ${need}."; break; }
    echo "Queue ${n} > ${threshold} (need ${need}); sleeping ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
  done
}

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
# Fresh seeds at the reported FOCUS-OT-L2 config, disjoint from the reported
# six, to characterize the bimodality found in the rebuttal runs.
#
# Pooled result over the first batch plus the gradpath and gradestimator
# pathwise arms (26 fresh seeds, all at the identical resolved config):
# mean 0.803 expert-normalized, 5 collapses below 0.5x = 19%
# (95% Wilson [0.085, 0.379]). The paper reports 0.905 from 6 seeds with no
# collapse; P(0 collapses in 6 | p=0.19) = 0.28, so that is an ordinary draw,
# not cherry-picking. SEEDSTAB_FILE=dflex_snu_humanoid_seedstability2.yaml adds
# 12 more disjoint seeds, which tightens the interval to [0.092, 0.334].
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
    --env_files "${SEEDSTAB_FILE:-dflex_snu_humanoid_seedstability.yaml}"
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

# --- F. isolating the simulator derivative (zDN4's Z3) -----------------------
# Table 13 compares FOCUS against ZO-FOCUS, which swaps the entire actor-critic
# for PPO/SAC -- and the audit showed it is not even reward-matched (C5), so it
# cannot isolate what the simulator derivatives contribute. This batch changes
# exactly one thing: the gradient estimator.
#
#   pathwise       = FOCUS as published, return differentiated through the sim
#   score_function = identical objective, reward, critic, network and horizon;
#                    REINFORCE with the critic as baseline
#
# Note simply detaching observations is not an option: the objective depends on
# the policy only through the dynamics, so the actor would get no gradient at
# all. Both arms run fresh so the comparison is not confounded by code or
# hardware drift.
if [[ " $BATCHES " == *" F "* ]]; then
  banner "F. gradient-estimator isolation  (2 envs x 2 arms x 6 seeds = 24 jobs)"
  for env in ${F_ENVS:-dflex_snu_humanoid dflex_hopper}; do
    steps="${MAX_STEPS[$env]}"
    for est in pathwise score_function; do
      echo "  ${env} / ${est}"
      cmd=(python spawner.py
        --task_name otil --docker
        --docker_image "${DOCKER_IMAGE:-/home/users/c/candidor/docker/mineral.sif}"
        --deployment slurm --runtime "$RUNTIME"
        --gpu_constraint "${GPU_CONSTRAINT:-COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE}"
        --no-cleanup --sweep --sweep_max 150 --base_algo SAPO
        --set "wandb.project=gradestimator_OTIL_SAPO-${env}-slurm"
        --set "agent.shac.max_agent_steps=${steps}"
        --set "agent.ckpt_every=${CKPT_EVERY}"
        --set "agent.otil.actor_grad_estimator=${est}"
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
        --env_files "${env}.yaml"
        --deploy_now)
      if [[ "$DRY_RUN" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; continue; fi
      wait_for_room "$BATCH_JOBS"
      before="$(job_count)"; "${cmd[@]}"; sleep "$SETTLE_SECONDS"; after="$(job_count)"
      delta=$(( after - before ))
      [[ "$delta" -gt 0 ]] && BATCH_JOBS="$delta"
    done
  done
fi

# --- U. unbalanced OT: relaxed marginals (Q9cS's fragmented plans) ----------
# Q9cS: "Some transport plans in Fig. 3(c) and Fig. 9(f)(h) show fragmented
# patterns, potentially caused by the uniform marginal constraints of OT, which
# that Open-end DTW could potentially avoid this issue."
#
# Balanced OT must move every unit of agent mass somewhere, so an agent state
# with no good expert match is still coupled to something. Relaxing the
# marginals with a KL penalty of strength rho lets mass go unmatched. Smaller
# rho = more relaxation; rho -> inf recovers balanced OT exactly (verified to
# 1.9e-8 in the unit test).
#
# No balanced arm is launched here: batch A's `pathwise` arm is already the
# reported FOCUS-OT-L2 configuration run fresh on all four envs, and the
# balanced code path is bit-identical before and after the unbalanced addition
# (verified: dloss/dLk/dgrad all 0.000e+00). Compare against that.
#
# NOTE this is double-edged. If relaxing the marginals clearly *improves*
# results, the published balanced choice is suboptimal in an obvious way; that
# is worth knowing before a reviewer suggests it, but report it plainly rather
# than burying it.
# 🔴 THE FIRST ATTEMPT (projects unbalancedot_OTIL_SAPO-*, 9 Nov 2026) IS
# INVALID -- do not use those runs. log_sinkhorn_unbalanced exponentiated the
# plan and only then normalised it; because scale = rho/(rho+eps) < 1 damps the
# Sinkhorn updates, log_u + log_v stopped offsetting log_K (-934 at eps=0.1)
# and exp() underflowed to a total mass of 3.5e-13 on part of each batch. The
# clamp_min(1e-12) then returned P * 1e12, leaving those elements at mass
# 0.34-0.36 instead of 1 and silently rescaling their per-step costs. All 24
# runs collapsed and the result was non-monotonic in rho, which is what gave it
# away. Normalising in the log domain fixes it (verified: mass exactly 1 for
# every rho >= 0.01, and rho -> inf recovers the balanced plan to 9.1e-7).
# The old "balanced path is bit-identical" test could never have caught this:
# it only exercises marginal_rho=None.
#
# The rho ladder is also wider now. The diagnostic showed rho=1.0 is already a
# STRONG relaxation at eps=0.1 (transport cost 9.14 vs 17.11 balanced), so the
# original {1.0, 0.1} bracketed only the aggressive end and never tested the
# reviewer's conjecture near the balanced limit. {100, 10, 1} spans
# near-balanced (max|P-P_bal| 6e-3) to moderate (0.33).
# U_TAG keeps the rerun in its own W&B projects.
if [[ " $BATCHES " == *" U "* ]]; then
  banner "U. unbalanced OT  (2 envs x ${n_rhos:-3} rho x 6 seeds)"
  for env in ${U_ENVS:-dflex_snu_humanoid dflex_humanoid}; do
    steps="${MAX_STEPS[$env]}"
    for rho in ${U_RHOS:-100 10 1}; do
      echo "  ${env} / rho=${rho}"
      cmd=(python spawner.py
        --task_name otil --docker
        --docker_image "${DOCKER_IMAGE:-/home/users/c/candidor/docker/mineral.sif}"
        --deployment slurm --runtime "$RUNTIME"
        --gpu_constraint "${GPU_CONSTRAINT:-COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE}"
        --no-cleanup --sweep --sweep_max 150 --base_algo SAPO
        --set "wandb.project=unbalancedot${U_TAG:+-$U_TAG}_OTIL_SAPO-${env}-slurm"
        --set "agent.shac.max_agent_steps=${steps}"
        --set "agent.ckpt_every=${CKPT_EVERY}"
        --set "agent.otil.loss_ot_marginal_rho=${rho}"
        --set "agent.otil.imitation_loss_type=ot"
        --set "agent.otil.loss_ot_cost_type=l2"
        --set "agent.otil.loss_best_of_k_k=8"
        --set "agent.otil.loss_best_of_k_tau=0.5"
        --set "agent.otil.loss_mlp_features_dim=64"
        --set "agent.otil.loss_ot_detach_plan=true"
        --set "agent.otil.actor_grad_estimator=pathwise"
        --set "agent.otil.critic_reward_mapping=exp"
        --set "agent.otil.loss_use_huber_speedup=false"
        --set "agent.otil.loss_use_detached_prev_obs=false"
        --set "agent.otil.critic_disabled=false"
        --set "agent.otil.critic_reward_shapping=true"
        --env_files "${env}.yaml"
        --deploy_now)
      if [[ "$DRY_RUN" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; continue; fi
      wait_for_room "$BATCH_JOBS"
      before="$(job_count)"; "${cmd[@]}"; sleep "$SETTLE_SECONDS"; after="$(job_count)"
      delta=$(( after - before )); [[ "$delta" -gt 0 ]] && BATCH_JOBS="$delta"
    done
  done
fi

echo
echo "Submitted. Projects to watch:"
echo "  gradpath_OTIL_SAPO-dflex_{hopper,ant,humanoid}-slurm"
echo "  seedstability_OTIL_SAPO-dflex_snu_humanoid-slurm"
echo "  rerun_OTIL_PPO-dflex_{snu_humanoid,humanoid}-slurm"
echo "  gradestimator_OTIL_SAPO-dflex_*-slurm                       (batch F)"
echo "  unbalancedot_OTIL_SAPO-dflex_{snu_humanoid,humanoid}-slurm  (batch U)"
echo
echo "Then: python plotter/rebuttal_plots.py --refresh"
