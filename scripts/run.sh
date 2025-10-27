


# DFlex

# # SAPO - Hopper
# python -m mineral.scripts.run \
# task=DFlex agent=DFlexAntSAPO task.env.env_name=hopper \
# logdir="workdir/DFlexHopper10M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
# agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
# agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
# agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
# wandb.mode=online wandb.project=rewarped \
# run=train_eval seed=130

# # SAPO - Ant
# python -m mineral.scripts.run \
# task=DFlex agent=DFlexAntSAPO task.env.env_name=ant \
# logdir="workdir/DFlexAnt10M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
# agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
# agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
# agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
# wandb.mode=online wandb.project=rewarped \
# run=train_eval seed=100

# # SAPO - Humanoid
# python -m mineral.scripts.run \
# task=DFlex agent=DFlexAntSAPO task.env.env_name=humanoid \
# logdir="workdir/DFlexHumanoid10M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
# agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
# agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
# agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
# agent.shac.critic_optim_kwargs.lr=5e-4 \
# agent.shac.target_critic_alpha=0.995 \
# wandb.mode=online wandb.project=rewarped \
# run=train_eval seed=110

# # SAPO - SNUHumanoid
# python -m mineral.scripts.run \
# task=DFlex agent=DFlexAntSAPO task.env.env_name=snu_humanoid \
# logdir="workdir/DFlexSNUHumanoid10M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
# agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
# agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
# agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
# agent.shac.critic_optim_kwargs.lr=5e-4 \
# agent.shac.target_critic_alpha=0.995 \
# wandb.mode=online wandb.project=rewarped \
# run=train_eval seed=120


# Rewarped
# # AntRun - SAPO
# python -m mineral.scripts.run \
# task=Rewarped agent=DFlexAntSAPO task.env.env_name=Ant task.env.env_suite=dflex \
# logdir="workdir/RewarpedAnt4M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
# agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
# agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
# agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
# wandb.mode=online wandb.project=rewarped \
# run=train_eval seed=1000

# HandReorient - SAPO
python -m mineral.scripts.run \
task=Rewarped agent=DFlexAntSAPO task.env.env_name=AllegroHand task.env.env_suite=isaacgymenvs \
logdir="workdir/RewarpedAllegroHand4M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
agent.shac.critic_optim_kwargs.lr=5e-4 \
agent.shac.target_critic_alpha=0.995 \
wandb.mode=online wandb.project=rewarped \
run=train_eval seed=1100

# # RollingFlat - SAPO
# python -m mineral.scripts.run \
# task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
# logdir="workdir/RewarpedRollingPin4M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
# num_envs=32 \
# agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
# agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
# agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
# agent.shac.critic_optim_kwargs.lr=5e-4 \
# agent.shac.target_critic_alpha=0.995 \
# agent.shac.max_agent_steps=4.1e6 agent.shac.max_epochs=4000 \
# wandb.mode=online wandb.project=rewarped \
# run=train_eval seed=1200

# # SoftJumper - SAPO
# python -m mineral.scripts.run \
# task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=Jumper task.env.env_suite=gradsim \
# logdir="workdir/RewarpedJumper6M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
# num_envs=32 \
# agent.network.encoder_kwargs.mlp_keys='com_q|com_qd|actions' \
# agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
# agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
# agent.shac.critic_optim_kwargs.lr=5e-4 \
# agent.shac.target_critic_alpha=0.995 \
# wandb.mode=online wandb.project=rewarped \
# run=train_eval seed=1300


# # HandFlip - SAPO
# python -m mineral.scripts.run \
# task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=Flip task.env.env_suite=dexdeform \
# logdir="workdir/RewarpedDexDeformFlip6M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
# num_envs=32 \
# agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
# agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
# agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
# agent.shac.critic_optim_kwargs.lr=5e-4 \
# agent.shac.target_critic_alpha=0.995 \
# wandb.mode=online wandb.project=rewarped \
# run=train_eval seed=1400


# FluidMove - SAPO
python -m mineral.scripts.run \
task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=Transport task.env.env_suite=softgym \
logdir="workdir/RewarpedSoftgymTransport4M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
num_envs=32 \
agent.network.encoder_kwargs.mlp_keys='com_q|joint_q|target_q' \
agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
agent.shac.critic_optim_kwargs.lr=5e-4 \
agent.shac.target_critic_alpha=0.995 \
agent.shac.max_agent_steps=4.1e6 agent.shac.max_epochs=4000 \
wandb.mode=online wandb.project=rewarped \
run=train_eval seed=1500