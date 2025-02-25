# Rewarped

## AntRun

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntPPO task.env.env_name=Ant task.env.env_suite=dflex \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedAnt4M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.ppo.max_agent_steps=4.1e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[128,64,32\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1000
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntSAC task.env.env_name=Ant task.env.env_suite=dflex \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedAnt4M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.sac.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1000
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntBPTT task.env.env_name=Ant task.env.env_suite=dflex \
    \
    logdir="workdir/RewarpedAnt4M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    agent.bptt.max_epochs=2000 agent.bptt.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1000
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntSHAC task.env.env_name=Ant task.env.env_suite=dflex \
    \
    logdir="workdir/RewarpedAnt4M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1000
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntSHAC2 task.env.env_name=Ant task.env.env_suite=dflex \
    \
    logdir="workdir/RewarpedAnt4M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1000
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntSAPO task.env.env_name=Ant task.env.env_suite=dflex \
    \
    logdir="workdir/RewarpedAnt4M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1000
</details>

## HandReorient

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntPPO task.env.env_name=AllegroHand task.env.env_suite=isaacgymenvs \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedAllegroHand4M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.ppo.max_agent_steps=4.1e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[512,256\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[256,256\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1100
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntSAC task.env.env_name=AllegroHand task.env.env_suite=isaacgymenvs \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedAllegroHand4M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.sac.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1100
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntBPTT task.env.env_name=AllegroHand task.env.env_suite=isaacgymenvs \
    \
    logdir="workdir/RewarpedAllegroHand4M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    agent.bptt.max_epochs=2000 agent.bptt.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1100
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntSHAC task.env.env_name=AllegroHand task.env.env_suite=isaacgymenvs \
    \
    logdir="workdir/RewarpedAllegroHand4M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1100
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntSHAC2 task.env.env_name=AllegroHand task.env.env_suite=isaacgymenvs \
    \
    logdir="workdir/RewarpedAllegroHand4M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1100
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=DFlexAntSAPO task.env.env_name=AllegroHand task.env.env_suite=isaacgymenvs \
    \
    logdir="workdir/RewarpedAllegroHand4M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1100
</details>

## RollingFlat

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperPPO task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedRollingPin4M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[512,256\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[256,256\] \
    \
    agent.ppo.max_agent_steps=4.1e6 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1200
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSAC task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedRollingPin4M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    agent.sac.max_agent_steps=4.1e6 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1200
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperBPTT task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
    \
    logdir="workdir/RewarpedRollingPin4M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    \
    agent.bptt.max_agent_steps=4.1e6 agent.bptt.max_epochs=4000 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1200
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSHAC task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
    \
    logdir="workdir/RewarpedRollingPin4M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    agent.shac.max_agent_steps=4.1e6 agent.shac.max_epochs=4000 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1200
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSHAC2 task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
    \
    logdir="workdir/RewarpedRollingPin4M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    agent.shac.max_agent_steps=4.1e6 agent.shac.max_epochs=4000 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1200
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=RollingPin task.env.env_suite=plasticinelab \
    \
    logdir="workdir/RewarpedRollingPin4M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    agent.shac.max_agent_steps=4.1e6 agent.shac.max_epochs=4000 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1200
</details>

## SoftJumper

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperPPO task.env.env_name=Jumper task.env.env_suite=gradsim \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedJumper6M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|com_qd|actions' \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[512,256\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[256,256\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1300
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSAC task.env.env_name=Jumper task.env.env_suite=gradsim \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedJumper6M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|com_qd|actions' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1300
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperBPTT task.env.env_name=Jumper task.env.env_suite=gradsim \
    \
    logdir="workdir/RewarpedJumper6M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|com_qd|actions' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1300
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSHAC task.env.env_name=Jumper task.env.env_suite=gradsim \
    \
    logdir="workdir/RewarpedJumper6M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|com_qd|actions' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1300
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSHAC2 task.env.env_name=Jumper task.env.env_suite=gradsim \
    \
    logdir="workdir/RewarpedJumper6M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|com_qd|actions' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1300
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=Jumper task.env.env_suite=gradsim \
    \
    logdir="workdir/RewarpedJumper6M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|com_qd|actions' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1300
</details>

## HandFlip

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperPPO task.env.env_name=Flip task.env.env_suite=dexdeform \
    task.env.no_grad=True \
    \
    logdir="workdir/DexDeformFlip6M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[512,256\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[256,256\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1400
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSAC task.env.env_name=Flip task.env.env_suite=dexdeform \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedDexDeformFlip6M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1400
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperBPTT task.env.env_name=Flip task.env.env_suite=dexdeform \
    \
    logdir="workdir/RewarpedDexDeformFlip6M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1400
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSHAC task.env.env_name=Flip task.env.env_suite=dexdeform \
    \
    logdir="workdir/RewarpedDexDeformFlip6M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1400
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSHAC2 task.env.env_name=Flip task.env.env_suite=dexdeform \
    \
    logdir="workdir/RewarpedDexDeformFlip6M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1400
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=Flip task.env.env_suite=dexdeform \
    \
    logdir="workdir/RewarpedDexDeformFlip6M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1400
</details>

## FluidMove

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperPPO task.env.env_name=Transport task.env.env_suite=softgym \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedSoftgymTransport4M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q|target_q' \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[512,256\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[256,256\] \
    \
    agent.ppo.max_agent_steps=4.1e6 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1500
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSAC task.env.env_name=Transport task.env.env_suite=softgym \
    task.env.no_grad=True \
    \
    logdir="workdir/RewarpedSoftgymTransport4M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q|target_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    agent.sac.max_agent_steps=4.1e6 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1500
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperBPTT task.env.env_name=Transport task.env.env_suite=softgym \
    \
    logdir="workdir/RewarpedSoftgymTransport4M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q|target_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    \
    agent.bptt.max_agent_steps=4.1e6 agent.bptt.max_epochs=4000 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1500
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSHAC task.env.env_name=Transport task.env.env_suite=softgym \
    \
    logdir="workdir/RewarpedSoftgymTransport4M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q|target_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    agent.shac.max_agent_steps=4.1e6 agent.shac.max_epochs=4000 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1500
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSHAC2 task.env.env_name=Transport task.env.env_suite=softgym \
    \
    logdir="workdir/RewarpedSoftgymTransport4M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q|target_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    agent.shac.max_agent_steps=4.1e6 agent.shac.max_epochs=4000 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1500
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=Rewarped agent=RewarpedJumperSAPO task.env.env_name=Transport task.env.env_suite=softgym \
    \
    logdir="workdir/Exp12W-RewarpedSoftgymTransport4M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    num_envs=32 \
    agent.network.encoder_kwargs.mlp_keys='com_q|joint_q|target_q' \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    agent.shac.max_agent_steps=4.1e6 agent.shac.max_epochs=4000 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=1500
</details>
