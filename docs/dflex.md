# DFlex

## Hopper

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntPPO task.env.env_name=hopper \
    task.env.no_grad=True \
    \
    logdir="workdir/DFlexHopper10M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.ppo.max_agent_steps=10e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[128,64,32\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=130
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSAC task.env.env_name=hopper \
    task.env.no_grad=True \
    \
    logdir="workdir/DFlexHopper10M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.sac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=130
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntBPTT task.env.env_name=hopper \
    \
    logdir="workdir/DFlexHopper10M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    agent.bptt.max_epochs=5000 agent.bptt.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=130
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSHAC task.env.env_name=hopper \
    \
    logdir="workdir/DFlexHopper10M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=130
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSHAC2 task.env.env_name=hopper \
    \
    logdir="workdir/DFlexHopper10M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=130
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSAPO task.env.env_name=hopper \
    \
    logdir="workdir/DFlexHopper10M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=130
</details>

## Ant

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntPPO task.env.env_name=ant \
    task.env.no_grad=True \
    \
    logdir="workdir/DFlexAnt10M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.ppo.max_agent_steps=10e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[128,64,32\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=100
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSAC task.env.env_name=ant \
    task.env.no_grad=True \
    \
    logdir="workdir/DFlexAnt10M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.sac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=100
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntBPTT task.env.env_name=ant \
    \
    logdir="workdir/DFlexAnt10M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    agent.bptt.max_epochs=5000 agent.bptt.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=100
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSHAC task.env.env_name=ant \
    \
    logdir="workdir/DFlexAnt10M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=100
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSHAC2 task.env.env_name=ant \
    \
    logdir="workdir/DFlexAnt10M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=100
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSAPO task.env.env_name=ant \
    \
    logdir="workdir/DFlexAnt10M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=100
</details>

## Humanoid

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntPPO task.env.env_name=humanoid \
    task.env.no_grad=True \
    \
    logdir="workdir/DFlexHumanoid10M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.ppo.max_agent_steps=10e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[256,128\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[128,128\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=110
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSAC task.env.env_name=humanoid \
    task.env.no_grad=True \
    \
    logdir="workdir/DFlexHumanoid10M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.sac.max_agent_steps=10e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[256,128\] \
    agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[128,128\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=110
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntBPTT task.env.env_name=humanoid \
    \
    logdir="workdir/DFlexHumanoid10M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    agent.bptt.max_epochs=5000 agent.bptt.max_agent_steps=10e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[256,128\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=110
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSHAC task.env.env_name=humanoid \
    \
    logdir="workdir/DFlexHumanoid10M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[256,128\] \
    agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[128,128\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=110
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSHAC2 task.env.env_name=humanoid \
    \
    logdir="workdir/DFlexHumanoid10M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[256,128\] \
    agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[128,128\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=110
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSAPO task.env.env_name=humanoid \
    \
    logdir="workdir/DFlexHumanoid10M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[128,64,32\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[64,64\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=110
</details>

## SNUHumanoid

<details>
<summary>PPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntPPO task.env.env_name=snu_humanoid \
    task.env.no_grad=True \
    \
    logdir="workdir/DFlexSNUHumanoid10M-PPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.ppo.max_agent_steps=10e6 \
    \
    agent.network.actor_critic_kwargs.mlp_kwargs.units=\[512,256\] \
    +agent.network.actor_critic_kwargs.critic_mlp_kwargs.units=\[256,256\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=120
</details>

<details>
<summary>SAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSAC task.env.env_name=snu_humanoid \
    task.env.no_grad=True \
    \
    logdir="workdir/DFlexSNUHumanoid10M-SAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.sac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    +agent.sac.target_entropy_scalar=0.5 \
    agent.sac.tau=0.005 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=120
</details>

<details>
<summary>APG / BPTT</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntBPTT task.env.env_name=snu_humanoid \
    \
    logdir="workdir/DFlexSNUHumanoid10M-BPTT/$(date +%Y%m%d-%H%M%S)" \
    agent.bptt.max_epochs=5000 agent.bptt.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=120
</details>

<details>
<summary>SHAC</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSHAC task.env.env_name=snu_humanoid \
    \
    logdir="workdir/DFlexSNUHumanoid10M-SHAC/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=120
</details>

<details>
<summary>SHAC2</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSHAC2 task.env.env_name=snu_humanoid \
    \
    logdir="workdir/DFlexSNUHumanoid10M-SHAC2/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=120
</details>

<details>
<summary>SAPO</summary>
    <br>

    python -m mineral.scripts.run \
    task=DFlex agent=DFlexAntSAPO task.env.env_name=snu_humanoid \
    \
    logdir="workdir/DFlexSNUHumanoid10M-SAPO/$(date +%Y%m%d-%H%M%S.%2N)" \
    agent.shac.max_epochs=5000 agent.shac.max_agent_steps=10e6 \
    \
    agent.network.actor_kwargs.mlp_kwargs.units=\[512,256\] \
    agent.network.critic_kwargs.mlp_kwargs.units=\[256,256\] \
    agent.shac.critic_optim_kwargs.lr=5e-4 \
    agent.shac.target_critic_alpha=0.995 \
    \
    wandb.mode=online wandb.project=rewarped \
    run=train_eval seed=120
</details>
