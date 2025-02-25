# mineral

A minimal(ish) reinforcement learning library that aggregates reliable implementations for:

- PPO, Proximal Policy Optimization ([`minimal-stable-PPO`](https://github.com/ToruOwO/minimal-stable-PPO), [`rl_games`](https://github.com/Denys88/rl_games)) #rl

- DDPG, Deep Deterministic Policy Gradient ([`pql`](https://github.com/Improbable-AI/pql), [`drqv2`](https://github.com/facebookresearch/drqv2), [`TD3`](https://github.com/sfujim/TD3)) #rl

- SAC, Soft Actor Critic ([`pql`](https://github.com/Improbable-AI/pql), [`pytorch-sac-ae`](https://github.com/denisyarats/pytorch_sac_ae)) #rl

- APG / BPTT, Analytic Policy Gradient / Backpropagation Through Time ([`DiffRL`](https://github.com/NVlabs/DiffRL)) #rl, #diffsim

- SHAC, Short Horizon Actor Critic ([`DiffRL`](https://github.com/NVlabs/DiffRL)) #rl, #diffsim

- SAPO, Soft Analytic Policy Optimization (`ours`) #rl, #diffsim

- BC, Behavioral Cloning #il

- DAPG, Demo Augmented Policy Gradient ([`maniskill2-learn`](https://github.com/haosulab/ManiSkill2-Learn)) #il, #rl, #off2on

#### Tags

| tag | description |
| --- | --- |
| #rl | (online) reinforcement learning |
| #offrl | offline reinforcement learning |
| #il | (offline) imitation learning |
| #off2on | offline-to-online |
| #diffsim | differentiable simulation |
| #mpc | model predictive control |

# Setup

```bash
conda create -n mineral python=3.10
conda activate mineral

pip install "torch>=2" torchvision
pip install git+https://github.com/etaoxing/mineral

# Rewarped
# see https://github.com/rewarped/rewarped

# DFlex
pip install git+https://github.com/rewarped/DiffRL
# make sure to run this so DFlex kernels are built
python -m dflex.examples.test_env --env AntEnv --num-envs 4 --render

# IsaacGymEnvs
# NOTE: requires python3.8
tar -xvf IsaacGym_Preview_4_Package.tar.gz
ln -s <path/to/isaacgym> thirdparty/isaacgym
cd third_party/isaacgym
pip install . --no-dependencies
cd -
git clone https://github.com/isaac-sim/IsaacGymEnvs third_party/IsaacGymEnvs
cd third_party/IsaacGymEnvs
git checkout b6dd437e68f94255f5a6306da76f2f0f9a634d6e
# comment out rl_games in INSTALL_REQUIRES of setup.py
pip install .
cd -
# ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
# use `import_isaacgym()` in mineral/envs/isaacgymenvs.py
```
