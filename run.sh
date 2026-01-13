python -m mineral.scripts.run \
    task=DFlex \
    task.env.env_name=ant \
    agent=ILD/DFlexAntSHAC2 \
    agent.shac.max_epochs=50000 \
    agent.shac.max_agent_steps=1e+08 \
    agent.shac.horizon_len=32 \
    agent.ild.demos.path=experts/demos/DFlex_ant_demos128_return9318_len1000.pt \
    agent.ild.demos.n_trajs=8 \
    agent.network.actor_kwargs.mlp_kwargs.units=[128,64,32] \
    agent.network.critic_kwargs.mlp_kwargs.units=[64,64] \
    logdir=workdir/DFlexAnt10M-ILD_20260113_003705/sweep_27 \
    wandb.mode=offline \
    wandb.project=temp-ild \
    run=train_eval \
    seed=120 \
    max_runtime=3h30m