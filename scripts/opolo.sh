set -e

python -m mineral.scripts.run \
task=DFlex agent=DFlexAntOPOLO task.env.env_name=hopper \
logdir="workdir_check/DFlexHopper10M-OPOLO/$(date +%Y%m%d-%H%M%S.%2N)" \
agent.sac.max_epochs=5 agent.sac.max_agent_steps=10e6 \
agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
wandb.mode=offline wandb.project=test-otil-slurm \
run=train_eval seed=130