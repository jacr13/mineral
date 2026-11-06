#!/usr/bin/env bash
# Expert-vs-learner videos for the rebuttal (reviewers zDN4 "no video of the
# experiment" and LMQH "expert-learner rollout comparisons, especially on SNU
# Humanoid").
#
# Needs an environment where dflex imports -- the project's py-310 env, or the
# container on the cluster. Run from the repo root:
#     bash plotter/make_rebuttal_videos.sh
#
# Checkpoints: the Hopper runs in grouped_exp_logdirs.yaml are available
# locally; Ant/Humanoid/SNU Humanoid reported checkpoints live on the cluster
# (their local logdirs hold logs only). Set LOGDIR_* below when running there.
set -euo pipefail

PY="${PY:-/home/joao/miniconda3/envs/py-310/bin/python}"
OUT="${OUT:-ICLR Reviews/videos}"
FRAMES="${FRAMES:-240}"
mkdir -p "$OUT"

D_HOPPER=experts/demos/DFlex_hopper_demos128_return4812_len1000.pt
D_ANT=experts/demos/DFlex_ant_demos128_return9318_len1000.pt
D_HUM=experts/demos/DFlex_humanoid_demos128_return8116_len984.pt
D_SNU=experts/demos/DFlex_snu_demos128_return6246_len929.pt

# --- expert reference, all four tasks (demonstrations are local) -------------
for pair in "hopper:$D_HOPPER" "ant:$D_ANT" "humanoid:$D_HUM" "snu_humanoid:$D_SNU"; do
  env="${pair%%:*}"; demos="${pair#*:}"
  "$PY" plotter/render_rollout.py --source demo --env "$env" --demos "$demos" \
      --frames "$FRAMES" --out "$OUT/${env}_expert.mp4"
done

# --- Hopper: expert vs each reported FOCUS variant, and the ILD baseline -----
declare -A HOPPER_RUNS=(
  [focus_l2]=plotter/workdir/DFlexHopper10M-OTIL_SHAC_20260121_034126/sweep_3
  [focus_ot_l2]=plotter/workdir/DFlexHopper10M-OTIL_SHAC_20260121_011014/sweep_19
  [focus_ot_cos]=plotter/workdir/DFlexHopper10M-OTIL_SHAC_20260121_011014/sweep_3
  [ild]=plotter/workdir/DFlexHopper10M-ILD_20260113_132856/sweep_18
)
declare -A HOPPER_LABELS=(
  [focus_l2]="FOCUS-L2"
  [focus_ot_l2]="FOCUS-OT-L2"
  [focus_ot_cos]="FOCUS-OT-Cos"
  [ild]="ILD (baseline)"
)
for name in "${!HOPPER_RUNS[@]}"; do
  "$PY" plotter/render_rollout.py --side-by-side --env hopper \
      --demos "$D_HOPPER" --logdir "${HOPPER_RUNS[$name]}" \
      --label "${HOPPER_LABELS[$name]}" \
      --frames "$FRAMES" --out "$OUT/hopper_expert_vs_${name}.mp4"
done

# --- On the cluster, add the other three tasks the same way ------------------
# for env/demos/logdir in the reported runs of grouped_exp_logdirs.yaml:
#   "$PY" plotter/render_rollout.py --side-by-side --env snu_humanoid \
#       --demos "$D_SNU" --logdir <reported logdir> \
#       --frames "$FRAMES" --out "$OUT/snu_humanoid_expert_vs_focus_ot_l2.mp4"

echo "videos written to $OUT"
