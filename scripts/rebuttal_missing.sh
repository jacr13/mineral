#!/usr/bin/env bash
# The runs the rebuttal tables still lack. Run from the repo on the cluster.
#
#   DRY_RUN=1 bash scripts/rebuttal_missing.sh        # print commands, submit nothing
#   bash scripts/rebuttal_missing.sh                  # both batches, 48 jobs
#   BATCHES="K" bash scripts/rebuttal_missing.sh      # only the L2 K=1 runs
#   K_ENVS="dflex_ant dflex_snu_humanoid" BATCHES="K" bash scripts/rebuttal_missing.sh
#
# K. FOCUS-L2 (no alignment) at K=1, four tasks x 6 seeds = 24 jobs.
#    Every FOCUS-L2 run on record uses K=8, so there is no evidence for the
#    claim that best-of-K matters most when no OT is used. The K=8 side exists
#    for Humanoid and SNU (batch V of rebuttal_overnight.sh); batch H below
#    supplies it for Hopper and Ant.
#
# H. FOCUS-L2 and FOCUS-OT-Cos at K=8 on Hopper and Ant, 2 x 2 x 6 = 24 jobs.
#    Fills the empty Hopper/Ant cells of the alignment table (reply to Q9cS)
#    and gives the fidelity table all three variants on those tasks.
#
# Settings mirror batch V of rebuttal_overnight.sh exactly. Each variant keeps
# its own reward mapping and feature map, as in the reported runs:
#   FOCUS-L2      loss=l2, 64-d feature map, exp
#   FOCUS-OT-Cos  loss=ot, cosine cost, no feature map, log_exp
# Seeds and input features come from tasks/otil/<env>.yaml (six seeds:
# 100 110 120 1000 1100 1200).
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-online}"   # set to offline if nodes cannot reach wandb
RUNTIME="${RUNTIME:-12h}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
MAX_JOBS="${MAX_JOBS:-100}"
DRY_RUN="${DRY_RUN:-0}"
BATCHES="${BATCHES:-K H}"
K_ENVS="${K_ENVS:-dflex_hopper dflex_ant dflex_humanoid dflex_snu_humanoid}"
K_VALUES="${K_VALUES:-1}"
H_ENVS="${H_ENVS:-dflex_hopper dflex_ant}"
H_VARIANTS="${H_VARIANTS:-l2 ot_cos}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
BATCH_JOBS="${BATCH_JOBS:-6}"
USER_NAME="${USER_NAME:-candidor}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$HERE")"

declare -A MAX_STEPS=(
  [dflex_hopper]=10000000 [dflex_ant]=10000000
  [dflex_humanoid]=15000000 [dflex_snu_humanoid]=15000000
)

job_count() { squeue -h -u "$USER_NAME" | wc -l | tr -d ' '; }
wait_for_room() {
  local need="$1"; local threshold=$(( MAX_JOBS - need ))
  (( threshold < 0 )) && threshold=0
  while true; do
    local n; n="$(job_count)"
    [[ "$n" -le "$threshold" ]] && break
    echo "Queue ${n} > ${threshold} (need ${need}); sleeping ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
  done
}

# submit <env> <variant: l2|ot_cos> <K> <wandb project tag>
submit() {
  local env="$1" variant="$2" k="$3" tag="$4"
  local loss cost mlp mapping
  if [[ "$variant" == "ot_cos" ]]; then
    loss=ot; cost=cosine; mlp=null; mapping=log_exp
  else
    loss=l2; cost=l2; mlp=64; mapping=exp
  fi
  echo "  ${env} / ${variant} / K=${k}"
  local cmd=(python spawner.py
    --task_name otil --docker
    --docker_image "${DOCKER_IMAGE:-/home/users/c/candidor/docker/mineral.sif}"
    --deployment slurm --runtime "$RUNTIME"
    --gpu_constraint "${GPU_CONSTRAINT:-COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE}"
    --no-cleanup --sweep --sweep_max 150 --base_algo SAPO
    --set "wandb.project=${tag}_OTIL_SAPO-${env}-slurm"
    --set "wandb.mode=${WANDB_MODE}"
    --set "agent.shac.max_agent_steps=${MAX_STEPS[$env]}"
    --set "agent.ckpt_every=${CKPT_EVERY}"
    --set "agent.otil.imitation_loss_type=${loss}"
    --set "agent.otil.loss_ot_cost_type=${cost}"
    --set "agent.otil.loss_mlp_features_dim=${mlp}"
    --set "agent.otil.loss_best_of_k_tau=0.5"
    --set "agent.otil.loss_best_of_k_k=${k}"
    --set "agent.otil.loss_ot_detach_plan=true"
    --set "agent.otil.critic_reward_mapping=${mapping}"
    --set "agent.otil.loss_use_huber_speedup=false"
    --set "agent.otil.loss_use_detached_prev_obs=false"
    --set "agent.otil.critic_disabled=false"
    --set "agent.otil.critic_reward_shapping=true"
    --env_files "${env}.yaml"
    --deploy_now)
  if [[ "$DRY_RUN" == "1" ]]; then printf '%q ' "${cmd[@]}"; echo; return; fi
  wait_for_room "$BATCH_JOBS"
  local before after delta
  before="$(job_count)"; "${cmd[@]}"; sleep "$SETTLE_SECONDS"; after="$(job_count)"
  delta=$(( after - before )); [[ "$delta" -gt 0 ]] && BATCH_JOBS="$delta"
  return 0
}

if [[ " $BATCHES " == *" K "* ]]; then
  echo "== K. FOCUS-L2 at K in {${K_VALUES}}"
  for env in $K_ENVS; do
    for k in $K_VALUES; do
      submit "$env" l2 "$k" "l2k"
    done
  done
fi

if [[ " $BATCHES " == *" H "* ]]; then
  echo "== H. FOCUS-L2 and FOCUS-OT-Cos at K=8 on Hopper and Ant"
  for env in $H_ENVS; do
    for variant in $H_VARIANTS; do
      submit "$env" "$variant" 8 "variants"
    done
  done
fi

echo "Done. Runs land under workdir/ as DFlex*-OTIL-sweep-critic-reward_<timestamp>."
