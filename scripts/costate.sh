#!/usr/bin/env bash
set -euo pipefail

python spawner.py \
    --task_name costate \
    --no-docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment local \
    --runtime 6h \
    --no-cleanup \
    --sweep \
    --sweep_max 150 \
    --base_algo SAPO \
    --set "agent.shac.max_epochs=10000" \
    --set "agent.shac.max_agent_steps=100000000" \
    --set "wandb.project=costate_local" \
    --env_files "dflex_ant.yaml" \
    --deploy_now
