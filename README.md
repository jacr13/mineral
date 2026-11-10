# FOCUS

First-Order Control Using Optimal Transport Supervision for Imitation from Observations.

This repository contains the implementation of FOCUS, a first-order imitation-from-observations method for contact-rich continuous control. FOCUS trains policies from state-only demonstrations by matching short differentiable rollouts to local expert trajectory segments with optimal transport supervision. The resulting local imitation costs are converted into pseudo-rewards and coupled with critic bootstrapping, so short-horizon differentiable updates can influence longer-horizon behavior without requiring temporally aligned full demonstrations or full-horizon backpropagation.

## Method

FOCUS is implemented as a differentiable actor-critic agent on top of the SHAC/SAPO training infrastructure:

- The agent rolls out the current policy for a short differentiable horizon.
- The rollout observations are compared against multiple state-only expert windows.
- A best-of-k local imitation objective selects the most compatible expert segment.
- The default objective uses entropy-regularized optimal transport via Sinkhorn iterations.
- Per-step OT costs are mapped to pseudo-rewards and used for critic targets.
- The actor receives first-order differentiable supervision from the local imitation objective plus bootstrapped value estimates.

Important implementation files:

- `mineral/agents/diffil/focus.py`: FOCUS training loop, pseudo-reward construction, critic bootstrapping, and optional OT-plan diagnostics.
- `mineral/agents/diffil/best_of_k.py`: best-of-k segment matching, Sinkhorn OT criterion, aligned regression, and cosine criteria.
- `mineral/cfgs/agent/FOCUS/`: reusable Hydra agent configs.
- `tasks/focus/`: experiment configs for the DFlex and Rewarped benchmark tasks.
- `mineral/common/demos.py`: demonstration loading utilities.

## Repository Layout

```text
mineral/
  agents/diffil/        FOCUS, ILD, and local matching losses
  agents/diffrl/        differentiable RL baselines such as SHAC and BPTT
  agents/dac/           adversarial imitation-from-observations baseline
  agents/gail/          GAIL-style baseline
  cfgs/                 Hydra task and agent defaults
  envs/                 DFlex, Rewarped, and IsaacGymEnv wrappers
tasks/
  focus/                FOCUS experiment entry points
  ild/ dac/ gail/       imitation baseline task files
  sapo/ shac/ bptt/     differentiable RL task files
scripts/                local and cluster launch helpers
```

## Setup

The main experiments use Python 3.12, PyTorch, Hydra, DFlex.

```bash
conda create -n focus python=3.12
conda activate focus
pip install poetry
poetry install
```

For direct `pip` use:

```bash
pip install -e .
pip install "torch>=2" torchvision
pip install gym==0.23.1
```

DFlex kernels need to be built once after installing DFlex:

```bash
python -m dflex.examples.test_env --env AntEnv --num-envs 4 --render
```

IsaacGymEnvs tasks require the Isaac Gym Preview 4 package and Python 3.8-compatible Isaac Gym dependencies. The DFlex tasks used by the main FOCUS experiments run with Python 3.12.

## Demonstrations

Task configs expect state-only demonstration tensors at paths such as:

```text
experts/demos/DFlex_hopper_demos128_return4812_len1000.pt
experts/demos/DFlex_ant_demos128_return9318_len1000.pt
experts/demos/DFlex_humanoid_demos128_return8116_len984.pt
experts/demos/DFlex_snu_demos128_return6246_len929.pt
```

## Running FOCUS

Single DFlex Hopper run:

```bash
python -m mineral.scripts.run \
  task=DFlex \
  task.env.env_name=hopper \
  agent=FOCUS/DFlexAntSHAC \
  agent.focus.demos.path=experts/demos/DFlex_hopper_demos128_return4812_len1000.pt \
  agent.focus.demos.n_trajs=8 \
  wandb.mode=disabled \
  logdir=workdir/DFlexHopper-FOCUS \
  run=train_eval \
  seed=100
```

Single DFlex Ant run with the OT settings used by the task config:

```bash
python -m mineral.scripts.run \
  task=DFlex \
  task.env.env_name=ant \
  agent=FOCUS/DFlexAntSHAC \
  agent.focus.demos.path=experts/demos/DFlex_ant_demos128_return9318_len1000.pt \
  agent.focus.demos.n_trajs=8 \
  agent.focus.imitation_loss_type=ot \
  agent.focus.loss_ot_cost_type=l2 \
  agent.focus.input_type=state_state \
  agent.focus.critic_reward_mapping=exp \
  wandb.mode=disabled \
  logdir=workdir/DFlexAnt-FOCUS \
  run=train_eval \
  seed=100
```

To launch from the task files through the spawner:

```bash
python spawner.py \
  --task_name focus \
  --env_files dflex_hopper.yaml \
  --deployment local \
  --no-docker \
  --set wandb.mode=disabled \
  --deploy_now
```

The main FOCUS task files are:

```text
tasks/focus/dflex_hopper.yaml
tasks/focus/dflex_ant.yaml
tasks/focus/dflex_humanoid.yaml
tasks/focus/dflex_snu_humanoid.yaml
```

## Baselines

Baseline task files use the same demonstration layout:

```bash
python spawner.py --task_name ild  --env_files dflex_hopper.yaml --deployment local --no-docker --set wandb.mode=disabled --deploy_now
python spawner.py --task_name dac  --env_files dflex_hopper.yaml --deployment local --no-docker --set wandb.mode=disabled --deploy_now
python spawner.py --task_name gail --env_files dflex_hopper.yaml --deployment local --no-docker --set wandb.mode=disabled --deploy_now
```

Use `--sweep --sweep_max N` to generate bounded sweeps from a task file's `sweep:` section.

## License

This repository is distributed under the license included in `LICENSE`.
