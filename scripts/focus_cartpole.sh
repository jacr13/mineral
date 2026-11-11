#!/usr/bin/env bash
# Tier 2: the non-periodic, order-sensitive task both reviewers asked for.
#
# Q9cS: "the paper does not evaluate non-periodic or order-sensitive tasks."
# zDN4: "local best-of-K matching ... may therefore fail on manipulation or
#        non-periodic multi-stage tasks."
#
# CartPole Swing-Up has two phases (recover from a perturbed state, then
# balance), so randomly sampled expert segments frequently come from a
# different phase than the agent occupies -- the predicted failure mode. This
# is a genuine test: FOCUS may lose here, and that is worth knowing before a
# reviewer asks.
#
# REQUIRES stochastic_init: True in tasks/{sapo,focus}/dflex_cartpole.yaml.
# The first launch (projects cartpole_SAPO-expert-slurm and
# cartpole_FOCUS_SAPO-slurm, 9 Nov 2026) inherited dflex's stochastic_init=False
# default, which starts every episode at the upright equilibrium: a zero-action
# policy scored exactly 0.00 there, the SAPO "expert" scored -316.6, and all
# five arms landed within seed noise of each other. Those runs are degenerate
# and must not be used; see section G of FOCUS_rebuttal_findings.md. TAG keeps
# the relaunch in separate W&B projects so the two cannot be pooled.
#
# Two stages. Stage 1 trains the expert; stage 2 cannot run until its
# demonstrations exist.
#
#   STAGE=expert bash scripts/focus_cartpole.sh     # 5 seeds, ~1h each
#   # then convert the best run's rollouts -- a plain cp does NOT work, the
#   # saved rollouts are per-episode dicts and the loader wants stacked arrays:
#   #   python experts/convert_rollouts.py <logdir>/demos/<file>.pt \
#   #     experts/demos/DFlex_cartpoleswingup_demos128.pt
#   STAGE=focus  bash scripts/focus_cartpole.sh     # 30 jobs
set -euo pipefail

STAGE="${STAGE:-focus}"
RUNTIME="${RUNTIME:-4h}"           # 240-step episodes; far cheaper than locomotion
CKPT_EVERY="${CKPT_EVERY:-2000}"
TAG="${TAG:-sto}"                  # W&B project suffix; "sto" = stochastic_init fix
DRY_RUN="${DRY_RUN:-0}"
DOCKER_IMAGE="${DOCKER_IMAGE:-/home/users/c/candidor/docker/mineral_uv.sif}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then printf '%q ' "$@"; echo; else "$@"; fi
}

if [[ "$STAGE" == "expert" ]]; then
  echo "STAGE 1: cartpole expert (SAPO), 5 seeds"
  run python spawner.py --task_name sapo --docker --docker_image "$DOCKER_IMAGE" \
    --deployment slurm --runtime "$RUNTIME" --gpu_constraint "$GPU_CONSTRAINT" \
    --no-cleanup --sweep --sweep_max 150 \
    --set "wandb.project=cartpole${TAG:+-$TAG}_SAPO-expert-slurm" \
    --set "wandb.mode=${WANDB_MODE:-online}" \
    --set "agent.ckpt_every=${CKPT_EVERY}" \
    --env_files dflex_cartpole.yaml --deploy_now
  echo
  echo "When finished, convert the best run's rollouts into demonstrations"
  echo "(a plain cp fails: the rollouts are per-episode dicts, not stacked):"
  echo "  python experts/convert_rollouts.py <logdir>/demos/<file>.pt \\"
  echo "    experts/demos/DFlex_cartpoleswingup_demos128.pt"
  exit 0
fi

if [[ ! -f experts/demos/DFlex_cartpoleswingup_demos128.pt && "$DRY_RUN" != "1" ]]; then
  echo "ERROR: experts/demos/DFlex_cartpoleswingup_demos128.pt is missing." >&2
  echo "Run STAGE=expert first, then install the demos (see above)." >&2
  exit 1
fi

# Which matching objective, holding everything else fixed. tau is per-row
# because the score scales differ by orders of magnitude; see the note in
# mineral/cfgs/agent/FOCUS/DFlexAntSAPO.yaml.
#        name            loss             cost    mlp   tau
ROWS=( "focus_ot_l2:ot:l2:64:0.5"
       "focus_ot_cos:ot:cosine:null:0.5"
       "focus_l2:l2:l2:64:0.5"
       "softdtw:softdtw:l2:64:0.5"
       "orca:orca:l2:64:200" )
ROWS_SEL="${ROWS_SEL:-0 1 2 3 4}"

echo "STAGE 2: FOCUS variants on cartpole  ($(wc -w <<< "$ROWS_SEL") rows x 6 seeds)"
for i in $ROWS_SEL; do
  IFS=':' read -r name loss cost mlp tau <<< "${ROWS[$i]}"
  echo "  row ${i}: ${name}  loss=${loss} cost=${cost} mlp=${mlp} tau=${tau}"
  run python spawner.py --task_name focus --docker --docker_image "$DOCKER_IMAGE" \
    --deployment slurm --runtime "$RUNTIME" --gpu_constraint "$GPU_CONSTRAINT" \
    --no-cleanup --sweep --sweep_max 150 --base_algo SAPO \
    --set "wandb.project=cartpole${TAG:+-$TAG}_FOCUS_SAPO-slurm" \
    --set "wandb.mode=${WANDB_MODE:-online}" \
    --set "agent.ckpt_every=${CKPT_EVERY}" \
    --set "agent.focus.imitation_loss_type=${loss}" \
    --set "agent.focus.loss_ot_cost_type=${cost}" \
    --set "agent.focus.loss_mlp_features_dim=${mlp}" \
    --set "agent.focus.loss_best_of_k_tau=${tau}" \
    --set "agent.focus.loss_best_of_k_k=8" \
    --set "agent.focus.critic_reward_mapping=exp" \
    --set "agent.focus.loss_use_huber_speedup=false" \
    --set "agent.focus.loss_use_detached_prev_obs=false" \
    --set "agent.focus.critic_disabled=false" \
    --set "agent.focus.critic_reward_shapping=true" \
    --env_files dflex_cartpole.yaml --deploy_now
done
echo
echo "Watch: cartpole${TAG:+-$TAG}_FOCUS_SAPO-slurm"
echo "NOTE: cartpole returns are negative; do not expert-normalise."
