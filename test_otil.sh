python -m mineral.scripts.run \
task=DFlex agent=DFlexAntOTIL task.env.env_name=ant \
logdir="workdir/DFlexAnt10M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
agent.otil.max_epochs=5000 agent.otil.max_agent_steps=10e6 \
agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
wandb.mode=online wandb.project=otil \
run=train_eval seed=100 \
agent.otil.demos_path="experts/DFlex_ant_demos64_return9340_len1000.pt"