# IsaacGymEnvs

## AllegroHand

<details>
<summary>PPO</summary>
    <br>

    PYTHONPATH=$PWD/third_party/isaacgym/python:$PWD/third_party/IsaacGymEnvs:$PYTHONPATH \
    python -m mineral.scripts.run \
    task=AllegroHand agent=AllegroHandPPO \
    logdir="workdir/AllegroHand-PPO/$(date +%Y%m%d-%H%M%S)"
</details>

<details>
<summary>DDPG</summary>
    <br>

    PYTHONPATH=$PWD/third_party/isaacgym/python:$PWD/third_party/IsaacGymEnvs:$PYTHONPATH \
    python -m mineral.scripts.run \
    task=AllegroHand agent=AllegroHandDDPG \
    logdir="workdir/AllegroHand-DDPG/$(date +%Y%m%d-%H%M%S)" \
    num_envs=4096 agent.ddpg.reward_shaper.scale=0.01
</details>

<details>
<summary>SAC</summary>
    <br>

    PYTHONPATH=$PWD/third_party/isaacgym/python:$PWD/third_party/IsaacGymEnvs:$PYTHONPATH \
    python -m mineral.scripts.run \
    task=AllegroHand agent=AllegroHandSAC \
    logdir="workdir/AllegroHand-SAC/$(date +%Y%m%d-%H%M%S)" \
    num_envs=4096 agent.sac.reward_shaper.scale=0.01
</details>
