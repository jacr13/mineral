#!/usr/bin/env bash
# Transport-plan gradient ablation: fixed-coupling surrogate vs differentiating
# through the Sinkhorn iterations.
#
# Why this exists. best_of_k.py computes the per-step costs as
# (P.detach() * C), so the actor gradient flows through the cost matrix only,
# at a frozen coupling. The paper states the opposite -- "differentiate through
# the costs, iterations, and simulated dynamics" (l. 288) and "gradients
# propagated through the Sinkhorn iterations" (l. 1058) -- which reviewer LMQH
# noticed. Measured on real normalized expert states, the two actor gradients
# agree to cos 0.96-0.99 for the cosine ground cost but only 0.54-0.80 for L2,
# the reported FOCUS-OT-L2 configuration, with the surrogate carrying ~55-80%
# of the magnitude. So this is not a wording slip for the headline variant and
# the behavioural question has to be answered with a run.
#
# Design. BOTH arms are run fresh, 6 seeds each. Comparing a new detach=false
# run against the numbers reported in January would confound the flag with
# everything else that changed since (code revision, GPU model, cluster load);
# running the matched detach=true arm alongside it removes that.
#
# Usage:
#   bash scripts/focus_gradpath.sh                      # SNU Humanoid (headline task)
#   ENVS="dflex_hopper" bash scripts/focus_gradpath.sh  # cheapest / fastest answer
#   ARMS="false" bash scripts/focus_gradpath.sh         # only the new arm
#   DRY_RUN=1 bash scripts/focus_gradpath.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

USER_NAME="${USER_NAME:-$(id -un)}"
POLL_SECONDS=60
MAX_JOBS="${MAX_JOBS:-100}"
SETTLE_SECONDS=5
# Reserve at least the sweep's maximum batch size. Queue deltas are unreliable:
# other jobs can finish or arrive while this batch is being submitted.
SEED_COUNT=6
BATCH_JOBS="${BATCH_JOBS:-6}"
DRY_RUN="${DRY_RUN:-0}"
CKPT_EVERY="${CKPT_EVERY:-2000}"

DOCKER_IMAGE="${DOCKER_IMAGE:-/home/users/c/candidor/docker/mineral_uv.sif}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE}"

# SNU Humanoid by default: it carries the headline result, and its L2 gradient
# gap (cos 0.57) is as large as Ant's. Hopper answers soonest at 8h.
ENVS="${ENVS:-dflex_snu_humanoid}"
ARMS="${ARMS:-true false}"   # loss_ot_detach_plan values

declare -A MAX_STEPS=(
  [dflex_hopper]=10000000 [dflex_ant]=10000000
  [dflex_humanoid]=15000000 [dflex_snu_humanoid]=15000000
)
declare -A RUNTIMES=(
  [dflex_hopper]=8h [dflex_ant]=8h
  [dflex_humanoid]=12h [dflex_snu_humanoid]=12h
)
# RUNTIME=12h overrides every env (the shared-gpu partition ceiling).
RUNTIME_OVERRIDE="${RUNTIME:-}"

fail() { echo "ERROR: $*" >&2; exit 1; }
read -r -a environments <<< "${ENVS//$'\n'/ }"
read -r -a arms <<< "${ARMS//$'\n'/ }"
(( ${#environments[@]} > 0 )) || fail "ENVS must contain at least one environment."
(( ${#arms[@]} > 0 )) || fail "ARMS must contain true and/or false."
for env in "${environments[@]}"; do
  [[ -n "${MAX_STEPS[$env]:-}" ]] || fail "Unsupported environment: ${env}"
done
for detach in "${arms[@]}"; do
  [[ "$detach" == "true" || "$detach" == "false" ]] || fail "Invalid arm: ${detach}; use true or false."
done
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN must be 0 or 1."
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || fail "MAX_JOBS must be a positive integer."
[[ "$BATCH_JOBS" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_JOBS must be a positive integer."
(( BATCH_JOBS >= SEED_COUNT )) || fail "BATCH_JOBS must reserve at least ${SEED_COUNT} slots."
(( MAX_JOBS >= BATCH_JOBS )) || fail "MAX_JOBS must be at least BATCH_JOBS (${BATCH_JOBS})."

n_envs=${#environments[@]}; n_arms=${#arms[@]}
echo "============================================================"
echo "ENVS : $ENVS"
echo "ARMS : loss_ot_detach_plan = $ARMS"
echo "PLAN : ${n_envs} env(s) x ${n_arms} arm(s) x ${SEED_COUNT} seeds = $(( n_envs * n_arms * SEED_COUNT )) jobs"
echo "============================================================"

job_count() { squeue -h -r -u "$USER_NAME" -o '%i' | wc -l | tr -d ' '; }
wait_for_room() {
  # $1 = how many jobs the next submission will add.
  local need="$1"
  local threshold=$(( MAX_JOBS - need ))
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

for env in "${environments[@]}"; do
  steps="${MAX_STEPS[$env]}"; runtime="${RUNTIME_OVERRIDE:-${RUNTIMES[$env]}}"
  echo "------------------------------------------------------------"
  echo "env=${env}  max_agent_steps=${steps}  runtime=${runtime}"
  echo "------------------------------------------------------------"
  for detach in "${arms[@]}"; do
    arm=$([[ "$detach" == "true" ]] && echo "fixed_coupling" || echo "through_plan")
    echo "  arm: ${arm}  (loss_ot_detach_plan=${detach})"

    # Everything below except loss_ot_detach_plan matches the reported
    # FOCUS-OT-L2 configuration recovered from W&B.
    cmd=(uv run --locked python spawner.py
      --task_name focus --docker --docker_image "${DOCKER_IMAGE}"
      --deployment slurm --runtime "${runtime}"
      --gpu_constraint "${GPU_CONSTRAINT}"
      --no-cleanup --sweep --sweep_max "$SEED_COUNT" --base_algo SAPO
      --set "wandb.project=gradpath_FOCUS_SAPO-${env}-slurm"
      --set "wandb.mode=${WANDB_MODE:-online}"
      --set "agent.shac.max_agent_steps=${steps}"
      --set "agent.ckpt_every=${CKPT_EVERY}"
      --set "agent.focus.loss_ot_detach_plan=${detach}"
      --set "agent.focus.imitation_loss_type=ot"
      --set "agent.focus.loss_ot_cost_type=l2"
      --set "agent.focus.loss_best_of_k_k=8"
      --set "agent.focus.loss_best_of_k_tau=0.5"
      --set "agent.focus.loss_mlp_features_dim=64"
      --set "agent.focus.critic_reward_mapping=exp"
      --set "agent.focus.loss_use_huber_speedup=false"
      --set "agent.focus.loss_use_detached_prev_obs=false"
      --set "agent.focus.critic_disabled=false"
      --set "agent.focus.critic_reward_shapping=true"
      --env_files "${env}.yaml"
      --deploy_now
    )
    if [[ "$DRY_RUN" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; continue; fi
    wait_for_room "$BATCH_JOBS"
    "${cmd[@]}"
    sleep "$SETTLE_SECONDS"
    after="$(job_count)"
    echo "Submitted ${arm}; queue now ${after}."
  done
done
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete; no jobs submitted."
else
  echo "Submitted. Compare the two arms within wandb project gradpath_FOCUS_SAPO-<env>-slurm."
fi
