#!/usr/bin/env bash
# Launches every prepared OOPS experiment on the SLURM cluster, cleanly, from one place.
#
# Parts (all use the fixed OOPS.update_net + the paper's network sizes; see tasks/oops_refnet/*.yaml):
#   hopper       hopper, 6 seeds, 12h, 1 update per env step (the paper's setting)                                    6 jobs x 12h
#   refnet       ant, humanoid, snu_humanoid: 6 seeds each, 12h, 1 update per env step                                18 jobs x 12h
#   utd          hopper update-ratio test (UTD 0.5 / 0.25 / 0.125, 3 seeds each) -- OPTIONAL, only needed if OOPS       9 jobs x 6h
#                should span the plotter's full 10M/15M step axis; the paper itself trains 1M steps at UTD 1
#
# PARTS is REQUIRED (there is deliberately no default): submitting a part twice duplicates its seeds in the same
# wandb group, and the plotter then counts them twice. Only launch parts you have not launched yet.
#
#   PARTS="hopper refnet" ./scripts/oops_all.sh           # all four environments (24 jobs)
#   PARTS="hopper" ./scripts/oops_all.sh                  # one part
#   DRY_RUN=1 PARTS="hopper refnet" ./scripts/oops_all.sh # only write the sbatch scripts under spawn/, submit nothing
set -euo pipefail

LOG_FILE="run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

DRY_RUN="${DRY_RUN:-0}"
SLURM_USER="candidor"
POLL_SECONDS=60
MAX_JOBS=100

if [[ -z "${PARTS:-}" ]]; then
  echo "ERROR: set PARTS to the part(s) to launch, e.g. PARTS=\"refnet\" ./scripts/oops_all.sh" >&2
  echo "       valid parts: hopper refnet utd (see the header of this script)." >&2
  echo "       No default on purpose: relaunching a part duplicates its seeds in the same wandb group." >&2
  exit 1
fi
read -r -a PART_LIST <<< "$PARTS"

# Each entry: "<task dir under tasks/>:<task file>:<number of jobs it submits>"
declare -a BATCHES=()
for part in "${PART_LIST[@]}"; do
  case "$part" in
    hopper)
      BATCHES+=("oops_refnet:dflex_hopper.yaml:6")
      ;;
    refnet)
      BATCHES+=("oops_refnet:dflex_ant.yaml:6" "oops_refnet:dflex_humanoid.yaml:6" "oops_refnet:dflex_snu_humanoid.yaml:6")
      ;;
    utd)
      BATCHES+=("oops_ablation:dflex_hopper_utd0.5.yaml:3" "oops_ablation:dflex_hopper_utd0.25.yaml:3" "oops_ablation:dflex_hopper_utd0.125.yaml:3")
      ;;
    *)
      echo "ERROR: unknown part '${part}' (valid: hopper refnet utd)" >&2
      exit 1
      ;;
  esac
done

job_count() {
  squeue -h -u "$SLURM_USER" | wc -l | tr -d ' '
}

wait_until_room_for() {
  local batch_jobs="$1"
  local threshold=$(( MAX_JOBS - batch_jobs ))
  if [[ "$threshold" -lt 0 ]]; then
    threshold=0
  fi

  while true; do
    local n
    n="$(job_count)"
    if [[ "$n" -le "$threshold" ]]; then
      echo "Queue size ${n} <= threshold ${threshold}. Room for another batch."
      break
    fi
    echo "Queue size ${n} > ${threshold}. Sleeping ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
  done
}

if [[ "$DRY_RUN" != "1" ]]; then
  git pull
fi

# Everything here needs the fixed OOPS.update_net (training used to normalize observations twice, acting only
# once). Refuse to submit from a checkout that lacks the fix, e.g. if it has not been pushed to this cluster's repo.
if ! grep -q "normalize twice during training" mineral/agents/oops/oops.py; then
  echo "ERROR: mineral/agents/oops/oops.py does not contain the double-normalization fix." >&2
  echo "       Commit/push it (and git pull on this machine) before launching." >&2
  exit 1
fi

# Fail before submitting anything if a task file is missing.
for batch in "${BATCHES[@]}"; do
  IFS=: read -r task_dir task_file _ <<< "$batch"
  if [[ ! -f "tasks/${task_dir}/${task_file}" ]]; then
    echo "ERROR: missing task file tasks/${task_dir}/${task_file}" >&2
    exit 1
  fi
done

SPAWN_FLAGS=(--no-cleanup --sweep --sweep_max 150)
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: writing sbatch scripts only, nothing will be submitted."
else
  SPAWN_FLAGS+=(--deploy_now)
fi

total_jobs=0
for batch in "${BATCHES[@]}"; do
  IFS=: read -r task_dir task_file batch_jobs <<< "$batch"

  echo "============================================================"
  echo "Launching ${task_dir}/${task_file} (${batch_jobs} jobs)"
  echo "============================================================"

  if [[ "$DRY_RUN" != "1" ]]; then
    wait_until_room_for "$batch_jobs"
  fi

  python spawner.py \
      --task_name "$task_dir" \
      --docker \
      --docker_image /home/users/c/candidor/docker/mineral.sif \
      --deployment slurm \
      "${SPAWN_FLAGS[@]}" \
      --env_files "$task_file"

  total_jobs=$(( total_jobs + batch_jobs ))
done

echo "Done: ${total_jobs} jobs $( [[ "$DRY_RUN" == "1" ]] && echo "written (dry run)" || echo "submitted" )."
