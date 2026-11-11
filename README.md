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
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync --locked
```

Run commands through `uv run`, or activate `.venv` with `source .venv/bin/activate`.
The source is run from the checkout; uv installs the dependencies without packaging
the project. The training entry point loads `.env` using `python-dotenv`.

Common dependency and task commands:

```bash
uv add <package>              # Add a dependency and update uv.lock
uv lock --upgrade            # Resolve newer versions within the declared ranges
uv sync --locked             # Install the versions recorded in uv.lock
uv sync --locked --no-dev     # Install runtime dependencies only
uv run poe format            # Run the Ruff tasks
```

For a local installation, DFlex builds its kernels on first import. The Docker
image already includes compiled kernels. To try the local DFlex example:

```bash
uv run --locked python -m dflex.examples.test_env --env AntEnv --num-envs 4 --render
```

IsaacGymEnvs tasks require the Isaac Gym Preview 4 package and Python 3.8-compatible Isaac Gym dependencies. The DFlex tasks used by the main FOCUS experiments run with Python 3.12.

### Docker

The CUDA development image uses uv to install Python and the locked dependencies,
including development tools. Its environment lives at `/opt/venv`, outside the
repository mounted at `/workspace`; `python` and `poe` work directly in the container.
DFlex's CPU and CUDA kernels compile during the image build, which does not require
a GPU. Running CUDA simulations still requires an NVIDIA GPU; CUDA initialization
and the driver's PTX compilation happen at runtime.

```bash
uv run poe docker_build  # Build candidj0/mineral:new with DFlex kernels included
uv run poe docker       # Start the container with GPU access
```

`uv run poe dbp` also pushes the image to Docker Hub. Rebuild existing Docker or
Apptainer images for the uv workflow. The spawner's `--uv_update` flag explicitly
upgrades dependencies before a job; otherwise generated commands use `uv.lock`.

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
uv run --locked python -m mineral.scripts.run \
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
uv run --locked python -m mineral.scripts.run \
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
uv run --locked python spawner.py \
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

## Running MENTOR

MENTOR (*Mimicking Expert states by iNverting the simulaTOR*, `mineral/agents/diffil/mentor.py`)
imitates state-only demonstrations produced by the same differentiable simulator:

1. **Invert:** recover the action behind each demonstration transition by Levenberg-Marquardt
   through one simulator step.
2. **Rejoin teacher:** from any state the learner visits, plan `teacher.horizon` open-loop actions
   in the simulator that bring the state back onto a demonstration, warm-started from the
   recovered actions of the nearest demonstration windows. The first action is the label.
3. **Distill:** DAgger. Behavioural cloning on the recovered actions, then per round learner
   rollouts, teacher labels, a cost gate and regression.

It needs no reward, critic or discriminator, and never reads the demonstrations' stored actions.
Observation-to-state maps exist for the DFlex Hopper, Ant and Humanoid
(`mineral/agents/diffil/mentor_sim.py`); Ant and Humanoid observations are used without their
trailing action coordinates.

```bash
uv run --locked python -m mineral.scripts.run \
  task=DFlex \
  task.env.env_name=hopper \
  agent=MENTOR/DFlex \
  agent.mentor.demos.path=experts/demos/DFlex_hopper_demos128_return4812_len1000.pt \
  logdir=workdir/DFlexHopper-MENTOR \
  run=train_eval \
  seed=100
```

Task files with the per-environment settings (gate, rollout resets, Stage I effort) are in
`tasks/mentor/`. Evaluation follows SHAC/FOCUS (two episodes per env, `num_actors * 2` episodes,
same `eval_scores/*` keys) with the deterministic policy. Besides `agent_steps` (learner env
steps), runs log `sim_steps/{learner,stage1,teacher_fwd,teacher_fwdbwd,total}` so budgets can
be compared in simulator transitions. GPU checks: `python -m pytest -q tests/test_mentor.py`.

## Baselines

Baseline task files use the same demonstration layout:

```bash
uv run --locked python spawner.py --task_name ild  --env_files dflex_hopper.yaml --deployment local --no-docker --set wandb.mode=disabled --deploy_now
uv run --locked python spawner.py --task_name dac  --env_files dflex_hopper.yaml --deployment local --no-docker --set wandb.mode=disabled --deploy_now
uv run --locked python spawner.py --task_name gail --env_files dflex_hopper.yaml --deployment local --no-docker --set wandb.mode=disabled --deploy_now
```

Use `--sweep --sweep_max N` to generate bounded sweeps from a task file's `sweep:` section.
An explicit `--runtime` overrides the task's `spawner.runtime`; without either,
the spawner defaults to 12 hours.

## License

This repository is distributed under the license included in `LICENSE`.

## Cluster supervisor exports

The spawner can export commit-pinned jobs into a cluster supervisor's data directory.
By default the supervisor keeps one checkout per branch on the cluster,
`$HOME/mcp_runs/<repo>_<branch>/`: it clones it on first use, fetches the branch when the
pinned commit is missing (also ahead of time while runs wait) and passes its path to the
job. The export records origin's URL (an SSH `git@host:` origin is recorded as HTTPS, so a
public repository clones without credentials) and the current branch; override them with
`--git_remote` and `--git_branch`. The commit must be pushed: `--enqueue` refuses a commit
that is not on `origin/<branch>`. Alternatively, `--cluster_repo` names an existing
checkout on the cluster that already contains the commit. Either way, jobs never pull a
moving branch or change the checkout: each job exports its commit into its own `source/`
directory and mounts that directory into the supplied Apptainer image. The image
provides the dependencies and must be compatible with the commit's `uv.lock`.

```bash
uv run --locked python spawner.py \
  --task_name focus --env_files dflex_humanoid.yaml \
  --deployment slurm --docker \
  --docker_image /cluster/images/mineral.sif \
  --cluster_input experts/demos=/cluster/datasets/mineral/demos \
  --git_commit HEAD \
  --supervisor_dir /home/joao/docker/cluster-supervisor/data \
  --sweep --sweep_max 6 --priority 40 --no-deploy_now
```

The paths above are examples; supply the actual cluster paths. `HEAD` is resolved
locally to a full commit hash. Uncommitted source changes are **not included**;
resolved task overrides and hyperparameters are included in `config.json` and in the
script. Exporting while developing is allowed and prints a reminder about this.

By default this creates inert `data/previews/<run-id>/` folders for review, plus
intermediate scripts in `data/generated/`. Nothing is submitted or registered.
Add `--enqueue` to generate and publish jobs to `data/incoming/`; the supervisor
registers immutable copies in `data/runs/<run-id>/`. Repeating the spawner command
creates new run IDs, including when switching from preview to enqueue. `--deploy_now`
is rejected in supervisor mode. Enqueued jobs remain waiting while paused or in dry-run;
turning submission on later makes them eligible, so use previews for review-only jobs.

Each exported folder contains:

```text
job.sbatch       # Complete Slurm script, pinned commit and training command
config.json      # Final spawner overrides, including seed and hyperparameters
run.json         # Stable supervisor ID, source revision, input paths and priority
out/             # Slurm stdout/stderr; present before Slurm opens these files
```

On the cluster, each job also creates `source/` and `results/` under its uploaded
run directory. Training outputs, checkpoints and metadata go to `results/`, separately
from Slurm logs in `out/`. The source snapshot is private to the run. `--cluster_input`
can be repeated for demonstrations or other files; inputs are mounted read-only for
Apptainer jobs. Untracked files such as `.env` are not copied automatically; if needed,
provide an explicit mapping or use your cluster's credential setup. Native jobs link
inputs and rely on host filesystem permissions instead of read-only container mounts.

The stable `CLUSTER_RUN_ID` exists before submission. `SLURM_JOB_ID` identifies the
cluster allocation afterward. Both, along with the pinned Git commit, are saved to
training metadata and W&B configuration by versions of this code containing that
support. Older pinned commits still have the supervisor-side `run.json` association.

Focused checks, requiring Python, PyYAML and pytest but no GPU or Slurm:

```bash
python -m pytest -q tests/test_cluster_jobs.py
```
