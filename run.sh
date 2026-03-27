# #!/usr/bin/env bash
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
		task=Rewarped \
		task.env.env_name=Jumper \
		task.env.env_suite=gradsim \
		agent=OTIL/RewarpedJumperSAPO \
		agent.network.encoder_kwargs.mlp_keys=\"com_q|com_qd|actions\" \
		agent.network.actor_kwargs.mlp_kwargs.units=[512,256] \
		agent.network.critic_kwargs.mlp_kwargs.units=[256,256] \
		agent.shac.critic_optim_kwargs.lr=5e-4 \
		agent.shac.target_critic_alpha=0.995 \
		agent.shac.max_epochs=100000 \
		agent.shac.max_agent_steps=1e+08 \
		agent.otil.demos.path=experts/demos/Rewarped_Jumper_demos64_return2046_len300.pt \
		agent.otil.demos.n_trajs=8 \
		agent.otil.loss_best_of_k_k=16 \
		agent.otil.critic_reward_mapping=log_exp \
		agent.otil.imitation_loss_type=ot \
		agent.otil.loss_ot_cost_type=cosine \
		agent.otil.loss_use_huber_speedup=false \
		agent.otil.loss_mlp_features_dim=null \
		agent.otil.loss_use_detached_prev_obs=true \
		logdir=workdir/RewarpedJumper6M-OTIL_20260205_162116/sweep_15 \
		num_envs=32 \
		wandb.mode=online \
		wandb.project=degug_softJumper_OTIL_--slurm-new \
		run=train_eval \
		seed=120 \
		max_runtime=12h"



# docker run \
#     --gpus=all \
#     --rm \
#     "${env_file[@]}" \
#     -it \
#     -u "$(id -u):$(id -g)" \
#     -v /etc/timezone:/etc/timezone:ro \
#     -v /etc/localtime:/etc/localtime:ro \
#     -v "$(pwd)":/workspace \
#     -w /workspace \
#     candidj0/mineral:latest \
#     bash -lc "python -m mineral.scripts.run \
#     task=Rewarped agent=SAPO/RewarpedJumper task.env.env_name=Jumper task.env.env_suite=gradsim \
#     \
#     logdir="workdir/RewarpedJumper6M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
#     num_envs=32 \
#     agent.network.encoder_kwargs.mlp_keys='com_q|com_qd|actions' \
#     \
#     agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
#     agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
#     agent.shac.critic_optim_kwargs.lr=5e-4 \
#     agent.shac.target_critic_alpha=0.995 \
#     \
#     wandb.mode=disabled wandb.project=rewarped \
#     run=train_eval seed=1300"



