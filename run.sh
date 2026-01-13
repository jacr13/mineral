#!/usr/bin/env bash
set -euo pipefail

env_file=()
if [ -f .env ]; then
  env_file=(--env-file .env)
fi

docker run \
    --gpus=all \
    --rm \
    "${env_file[@]}" \
    -it \
    -u "$(id -u):$(id -g)" \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v "$(pwd)":/workspace \
    -w /workspace \
    candidj0/mineral:latest \
    bash -lc "python -m mineral.scripts.run \
        task=DFlex \
        task.env.env_name=hopper \
        agent=DAC/DFlexAnt \
        agent.sac.max_agent_steps=10e6 \
        agent.dac.demos.path=experts/demos/DFlex_hopper_demos128_return4812_len1000.pt \
        agent.dac.demos.n_trajs=8 \
        agent.network.actor_kwargs.mlp_kwargs.units=[128,64,32] \
        agent.network.critic_kwargs.mlp_kwargs.units=[64,64] \
        logdir=workdir/DFlexHopper10M-DAC_\$(date +%Y%m%d-%H%M%S.%2N) \
        wandb.mode=online \
        wandb.project=test-rewarped-local \
        run=train_eval \
        seed=120 \
        max_runtime=3h30m"
