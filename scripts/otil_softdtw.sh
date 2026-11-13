#!/usr/bin/env bash
# FOCUS-softDTW sweep: the ordered-alignment comparison reviewers Q9cS and LMQH
# asked for ("add DTW-based baselines", "justify the necessity of OT"), plus the
# best-of-K temperature sweep that explains the flat K-curve in Fig. 13.
#
# Modelled on scripts/otil.sh. Every FOCUS-specific setting below is matched to
# the configuration the *reported* FOCUS-OT-L2 runs actually used (recovered
# from W&B -- see "ICLR Reviews/FOCUS_rebuttal_findings.md" section A.1):
#   critic_reward_mapping=exp, loss_ot_cost_type=l2, loss_mlp_features_dim=64,
#   loss_use_huber_speedup=false, loss_use_detached_prev_obs=false,
#   critic_disabled=false, K=8, H=32, 6 seeds, SAPO backbone.
# Only the alignment (or, for rows 4-5, the temperature) differs, so each row is
# a controlled substitution rather than a different method.
#
# Usage:
#   bash scripts/otil_softdtw.sh                  # all four tasks, all five rows
#   ENVS="dflex_hopper dflex_ant" bash scripts/otil_softdtw.sh   # just these two
#   ROWS="0 1" bash scripts/otil_softdtw.sh       # only the soft-DTW rows
#   DRY_RUN=1 bash scripts/otil_softdtw.sh        # print the commands, submit nothing
#
# It prints the full job plan and the env list before submitting anything.
set -euo pipefail

USER_NAME="${USER_NAME:-candidor}"
POLL_SECONDS=60
MAX_JOBS="${MAX_JOBS:-100}"
SETTLE_SECONDS=5
# Jobs added per submission. Seeded with the sweep's seed count and then
# corrected from the measured queue delta. It must never be set to the total
# queue size: with anything already queued that inflates it and every later
# batch waits for a threshold it may never reach.
BATCH_JOBS="${BATCH_JOBS:-6}"
DRY_RUN="${DRY_RUN:-0}"
# Epochs between intermediate checkpoints (default config: 100, unpruned).
CKPT_EVERY="${CKPT_EVERY:-2000}"
# STEPS overrides the per-env interaction budget (used for the extended-budget
# run that matches how far the reported baselines actually got).
STEPS_OVERRIDE="${STEPS:-}"

DOCKER_IMAGE="${DOCKER_IMAGE:-/home/users/c/candidor/docker/mineral.sif}"
# Keep off Pascal nodes (P100 / Titan X): scripts/otil.sh documents that a
# Humanoid rerun there got wall-clock-capped while Turing/Ampere seeds finished
# in half the time.
GPU_CONSTRAINT="${GPU_CONSTRAINT:-COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE}"

# All four tasks by default, humanoids first: that is the order to cancel in if
# compute runs short (SNU Humanoid and Humanoid carry the headline result), but
# the paper reports all four, so a comparison on two of them would be a gap a
# reviewer could reasonably pick at.
ENVS="${ENVS:-dflex_snu_humanoid dflex_humanoid dflex_hopper dflex_ant}"

declare -A MAX_STEPS=(
  [dflex_hopper]=10000000 [dflex_ant]=10000000
  [dflex_humanoid]=15000000 [dflex_snu_humanoid]=15000000
)
declare -A RUNTIMES=(
  [dflex_hopper]=8h [dflex_ant]=8h
  [dflex_humanoid]=12h [dflex_snu_humanoid]=12h
)
RUNTIME_OVERRIDE="${RUNTIME:-}"

# ---------------------------------------------------------------------------
# Sweep rows. Index-aligned; each is one batch of 6 seeds per env.
#
#  0  softdtw        K=8  -- does the alignment objective matter?        (Q1,Q2)
#  1  softdtw_subseq K=1  -- is best-of-K a sampled subsequence soft-DTW? (Q3)
#  2  softdtw_subseq K=8  -- subsequence alignment with the full pool
#  3  ot             K=8  tau=0.02 -- OT with a temperature matched to its
#                                     segment-score scale
#  4  ot (cosine)    K=8  tau=0.01 -- the Fig. 13 test: with beta on the right
#                                     scale, does K start to matter for Cos?
#
# Rows 3-4 test the mechanism in findings section E.2: at the fixed beta = 0.5
# the effective candidate count is ~1.2-2.0/8 for OT-L2 but ~6.5-7.4/8 for
# OT-Cos, i.e. best-of-K silently degenerates to averaging for the bounded
# cosine cost. These two rows are cheap and turn that observation into a result.
# ---------------------------------------------------------------------------
# The transport-plan gradient ablation lives in scripts/otil_gradpath.sh: it is
# a different question (how the gradient is taken, not which alignment is used)
# and needs its own wandb project so the provenance stays readable.
#  5  softdtw (cosine) K=8 -- soft-DTW at FOCUS-OT-Cos's operating point.
#     Rows 0-2 are controlled substitutions against FOCUS-OT-**L2**, which was
#     the headline arm when this sweep was written. FOCUS-OT-Cos is now the
#     strongest arm, and it differs from row 0 in three ways at once: cosine
#     ground cost, log_exp reward transform, and raw features (mlp=null)
#     instead of a 64-d MLP. So the existing table cannot separate "transport
#     matters" from "cosine cost + log reward + raw features matter" -- which
#     is exactly Reviewer Q9cS's objection. This row matches OT-Cos on all
#     three and changes only the alignment, so it answers that question in
#     whichever direction it falls.
ROW_NAME=(softdtw_k8 subseq_k1 subseq_k8 ot_l2_lowtau ot_cos_lowtau softdtw_cos)
ROW_LOSS=(softdtw softdtw_subseq softdtw_subseq ot ot softdtw)
ROW_K=(8 1 8 8 8 8)
ROW_TAU=(0.5 0.5 0.5 0.02 0.01 0.5)
ROW_COST=(l2 l2 l2 l2 cosine cosine)
ROW_MLP=(64 64 64 64 null null)
# Reward transform per row. Rows 0-4 keep the exp they were launched with, so
# their behaviour is unchanged; row 5 needs log_exp to match reported OT-Cos.
ROW_MAP=(exp exp exp exp exp log_exp)

ROWS="${ROWS:-0 1 2 3 4 5}"

n_envs=$(wc -w <<< "$ENVS"); n_rows=$(wc -w <<< "$ROWS")
echo "============================================================"
echo "ENVS : $ENVS"
echo "ROWS : $ROWS"
echo "PLAN : ${n_envs} env(s) x ${n_rows} row(s) x 6 seeds = $(( n_envs * n_rows * 6 )) jobs"
echo "============================================================"

job_count() { squeue -h -u "$USER_NAME" | wc -l | tr -d ' '; }

wait_for_room() {
  # $1 = how many jobs the next submission will add.
  local need="$1"
  local threshold=$(( MAX_JOBS - need ))
  (( threshold < 0 )) && threshold=0
  while true; do
    local n; n="$(job_count)"
    if [[ "$n" -le "$threshold" ]]; then
      echo "Queue ${n} <= ${threshold}; room for ${need} more."
      break
    fi
    echo "Queue ${n} > ${threshold} (need room for ${need}); sleeping ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
  done
}

for env in $ENVS; do
  steps="${MAX_STEPS[$env]}"
  runtime="${RUNTIME_OVERRIDE:-${RUNTIMES[$env]}}"
  echo "============================================================"
  echo "env=${env}  max_agent_steps=${steps}  runtime=${runtime}"
  echo "============================================================"

  for i in $ROWS; do
    name="${ROW_NAME[$i]}"
    echo "------------------------------------------------------------"
    echo "row ${i}: ${name}  loss=${ROW_LOSS[$i]} K=${ROW_K[$i]} tau=${ROW_TAU[$i]} cost=${ROW_COST[$i]} mlp=${ROW_MLP[$i]}"
    echo "------------------------------------------------------------"

    cmd=(python spawner.py
      --task_name otil
      --docker
      --docker_image "${DOCKER_IMAGE}"
      --deployment slurm
      --runtime "${runtime}"
      --gpu_constraint "${GPU_CONSTRAINT}"
      --no-cleanup
      --sweep
      --sweep_max 150
      --base_algo SAPO
      # NOTE: do not --set logdir here. spawner.py builds a per-variant
      # workdir/<experiment>_<timestamp>/sweep_<n> path (spawner.py:556-572) and
      # then applies CLI overrides on top, so overriding logdir makes all six
      # seeds share one directory. The inherited directory name still says
      # "sweep-critic-reward" (it comes from the task yaml's sweep.logdir);
      # the runs are identified by wandb.project and the timestamp instead.
      --set "wandb.project=softdtw_OTIL_SAPO-${env}-slurm"
      --set "wandb.mode=${WANDB_MODE:-online}"
      --set "agent.shac.max_agent_steps=${STEPS_OVERRIDE:-${steps}}"
      # Checkpoint volume: ckpt_every defaults to 100 epochs and nothing prunes
      # old files, so a run leaves 50-185 .pth files (20-78 MB). Across this
      # sweep that is tens of GB, and a disk-space purge is what destroyed the
      # reported runs' checkpoints. These runs are scored on returns, so keep
      # far fewer intermediate checkpoints; best/final/latest are still written.
      --set "agent.ckpt_every=${CKPT_EVERY}"
      --set "agent.otil.imitation_loss_type=${ROW_LOSS[$i]}"
      --set "agent.otil.loss_best_of_k_k=${ROW_K[$i]}"
      --set "agent.otil.loss_best_of_k_tau=${ROW_TAU[$i]}"
      --set "agent.otil.loss_ot_cost_type=${ROW_COST[$i]}"
      --set "agent.otil.loss_mlp_features_dim=${ROW_MLP[$i]}"
      --set "agent.otil.loss_dtw_gamma=0.1"
      --set "agent.otil.loss_dtw_normalize=rows"
      --set "agent.otil.critic_reward_mapping=${ROW_MAP[$i]}"
      --set "agent.otil.loss_use_huber_speedup=false"
      --set "agent.otil.loss_use_detached_prev_obs=false"
      --set "agent.otil.critic_disabled=false"
      --set "agent.otil.critic_reward_shapping=true"
      --env_files "${env}${ENV_SUFFIX:-}.yaml"
      --deploy_now
    )

    if [[ "$DRY_RUN" == "1" ]]; then
      printf '%q ' "${cmd[@]}"; echo; continue
    fi

    wait_for_room "$BATCH_JOBS"
    before="$(job_count)"
    "${cmd[@]}"
    sleep "$SETTLE_SECONDS"
    after="$(job_count)"
    delta=$(( after - before ))
    if [[ "$delta" -gt 0 ]]; then
      BATCH_JOBS="$delta"
      echo "This batch added ${delta} job(s); queue now ${after}."
    else
      echo "WARNING: queue did not grow after submission (before=${before}, after=${after})." >&2
    fi
  done
done

echo "All requested rows submitted."
