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
		task.env.env_name=ant \
		agent=OTIL/DFlexAntSHAC \
		agent.shac.max_epochs=50000 \
		agent.shac.max_agent_steps=1e+08 \
		agent.otil.demos.path=experts/demos/DFlex_ant_demos128_return9318_len1000.pt \
		agent.otil.demos.n_trajs=4 \
		agent.otil.critic_reward_mapping=exp \
		agent.otil.imitation_loss_type=ot \
		agent.otil.input_type=state_state \
		agent.otil.loss_mlp_features_dim=64 \
		agent.otil.loss_ot_cost_type=l2 \
		agent.otil.loss_use_huber_speedup=false \
		agent.otil.loss_use_detached_prev_obs=false \
		agent.network.actor_kwargs.mlp_kwargs.units=[128,64,32] \
		agent.network.critic_kwargs.mlp_kwargs.units=[64,64] \
		logdir=workdir/DFlexAnt10M-OTIL_SHAC_20260120_175806/sweep_6 \
		wandb.mode=online \
		wandb.project=OTIL_SHAC-dflex_ant-slurm-new \
		run=train_eval \
		seed=110 \
		max_runtime=4h"
