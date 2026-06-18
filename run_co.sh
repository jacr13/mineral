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
    candidj0/mineral:new \
    bash -lc "python -m mineral.scripts.run \
        task=DFlex \
        task.env.env_name=hopper \
        agent=Costate/DFlexAntSHAC \
        agent.network.actor_kwargs.mlp_kwargs.units=[128,64,32] \
        agent.network.critic_kwargs.mlp_kwargs.units=[64,64] \
        agent.shac.max_epochs=50000 \
        agent.shac.max_agent_steps=10000000 \
        agent.otil.demos.path=experts/demos/DFlex_hopper_demos128_return4812_len1000.pt \
        agent.otil.demos.n_trajs=8 \
        agent.otil.loss_best_of_k_k=8 \
        agent.otil.critic_reward_mapping=exp \
        agent.otil.imitation_loss_type=ot \
        agent.otil.loss_ot_cost_type=l2 \
        agent.otil.loss_use_huber_speedup=false \
        agent.otil.loss_mlp_features_dim=null \
        agent.otil.loss_use_detached_prev_obs=false \
        agent.costate.costate_grad_clip=null \
        agent.costate.costate_grad_normalize=false \
        agent.costate.loss_scale=10.0 \
        agent.costate.use_reparameterized_surrogate=true \
        logdir=workdir/DFlexHopper10M-Costate-debug/$(date +%Y%m%d-%H%M%S.%2N) \
        wandb.mode=online \
        wandb.project=debug_costate \
        run=train_eval \
        seed=130 \
        max_runtime=1h"
