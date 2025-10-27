set -e


# DFlex

# OTIL - Hopper
python -m mineral.scripts.run \
task=DFlex agent=DFlexAntOTIL task.env.env_name=hopper \
logdir="workdir_check/DFlexHopper10M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
agent.otil.max_epochs=5 agent.otil.max_agent_steps=10e6 \
agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
wandb.mode=offline wandb.project=OTIL \
run=train_eval seed=130 \
agent.otil.demos_path="experts/DFlex_hopper_demos64_return5437_len1000.pt"

# # OTIL - Ant
python -m mineral.scripts.run \
task=DFlex agent=DFlexAntOTIL task.env.env_name=ant \
logdir="workdir_check/DFlexAnt10M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
agent.otil.max_epochs=5 agent.otil.max_agent_steps=10e6 \
agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
wandb.mode=offline wandb.project=OTIL \
run=train_eval seed=100 \
agent.otil.demos_path="experts/DFlex_ant_demos64_return9340_len1000.pt"

# OTIL - Humanoid
python -m mineral.scripts.run \
task=DFlex agent=DFlexAntOTIL task.env.env_name=humanoid \
logdir="workdir_check/DFlexHumanoid10M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
agent.otil.max_epochs=5 agent.otil.max_agent_steps=10e6 \
agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
wandb.mode=offline wandb.project=OTIL \
run=train_eval seed=110 \
agent.otil.demos_path="experts/DFlex_humanoid_demos1_return8899_len1000.pt"

# OTIL - SNUHumanoid
python -m mineral.scripts.run \
task=DFlex agent=DFlexAntOTIL task.env.env_name=snu_humanoid \
logdir="workdir_check/DFlexSNUHumanoid10M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
agent.otil.max_epochs=5 agent.otil.max_agent_steps=10e6 \
agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
wandb.mode=offline wandb.project=OTIL \
run=train_eval seed=120 \
agent.otil.demos_path="experts/DFlex_snu_demos44_return6841_len1000.pt"


# Rewarped
# OTIL - AntRun
python -m mineral.scripts.run \
task=Rewarped agent=DFlexAntOTIL task.env.env_name=Ant task.env.env_suite=dflex \
logdir="workdir_check/RewarpedAnt4M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
agent.otil.max_epochs=5 agent.otil.max_agent_steps=4.1e6 \
agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
wandb.mode=offline wandb.project=OTIL \
run=train_eval seed=1000 \
agent.otil.demos_path=""

# OTIL - HandReorient
python -m mineral.scripts.run \
task=Rewarped agent=DFlexAntOTIL task.env.env_name=AllegroHand task.env.env_suite=isaacgymenvs \
logdir="workdir_check/RewarpedAllegroHand4M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
agent.otil.max_epochs=5 agent.otil.max_agent_steps=4.1e6 \
agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
wandb.mode=offline wandb.project=OTIL \
run=train_eval seed=1100 \
agent.otil.demos_path=""

# OTIL - RollingFlat
python -m mineral.scripts.run \
task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
logdir="workdir_check/RewarpedRollingPin4M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
num_envs=32 \
agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
agent.otil.max_agent_steps=4.1e6 agent.otil.max_epochs=5 \
wandb.mode=offline wandb.project=OTIL \
run=train_eval seed=1200 \
agent.otil.demos_path=""

# OTIL - SoftJumper
python -m mineral.scripts.run \
task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=Jumper task.env.env_suite=gradsim \
logdir="workdir_check/RewarpedJumper6M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
num_envs=32 \
agent.network.encoder_kwargs.mlp_keys='com_q|com_qd|actions' \
agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
wandb.mode=offline wandb.project=OTIL \
run=train_eval seed=1300 \
agent.otil.demos_path=""


# OTIL - HandFlip
python -m mineral.scripts.run \
task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=Flip task.env.env_suite=dexdeform \
logdir="workdir_check/RewarpedDexDeformFlip6M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
num_envs=32 \
agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
wandb.mode=offline wandb.project=OTIL \
run=train_eval seed=1400 \
agent.otil.demos_path=""


# # OTIL - FluidMove
# python -m mineral.scripts.run \
# task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=Transport task.env.env_suite=softgym \
# logdir="workdir_check/Exp12W-RewarpedSoftgymTransport4M-OTIL/$(date +%Y%m%d-%H%M%S.%2N)" \
# num_envs=32 \
# agent.network.encoder_kwargs.mlp_keys='com_q|joint_q|target_q' \
# agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
# agent.otil.max_agent_steps=4.1e6 agent.otil.max_epochs=5 \
# wandb.mode=offline wandb.project=OTIL \
# run=train_eval seed=1500 \
# agent.otil.demos_path=""
