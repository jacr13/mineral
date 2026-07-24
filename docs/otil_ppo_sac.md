# Detached OTIL reward baselines

`OTILPPO` and `OTILSAC` use the existing PPO and SAC updates without
backpropagating through the simulator or the OT objective. Each collection
window is scored with the same Best-of-K sequence objective used by FOCUS, and
the resulting detached per-step OT rewards are inserted into the normal PPO or
SAC buffer.

The Slurm sweep launchers are:

```bash
bash scripts/otilppo.sh
bash scripts/otilsac.sh
```

Their environment-specific definitions live under `tasks/otilppo/` and
`tasks/otilsac/`.

Each launcher submits four DFlex environments, the three paper conditions
(`FOCUS-L2`, `FOCUS-OT-L2`, and `FOCUS-OT-Cos`), and the six FOCUS seeds:
72 jobs in total. The FOCUS-only `critic_disabled` configuration is
intentionally excluded because PPO and SAC both require a critic.

The following commands reproduce the DFlex Ant setup with the same expert
file, horizon, K, state-transition representation, OT cost, reward mapping,
policy widths, and 10M environment-step budget:

```bash
python -m mineral.scripts.run \
  task=DFlex \
  task.env.env_name=ant \
  task.env.no_grad=true \
  agent=OTIL/DFlexAntPPO \
  agent.ppo.max_agent_steps=10000000 \
  agent.otil.demos.path=experts/demos/DFlex_ant_demos128_return9318_len1000.pt \
  agent.otil.input_type=state_state \
  agent.otil.loss_best_of_k_k=8 \
  agent.otil.loss_ot_cost_type=l2 \
  agent.otil.loss_use_huber_speedup=false \
  agent.otil.loss_use_detached_prev_obs=false \
  agent.otil.critic_reward_mapping=exp \
  agent.otil.critic_reward_shapping=true
```

```bash
python -m mineral.scripts.run \
  task=DFlex \
  task.env.env_name=ant \
  task.env.no_grad=true \
  agent=OTIL/DFlexAntSAC \
  agent.sac.max_agent_steps=10000000 \
  agent.otil.demos.path=experts/demos/DFlex_ant_demos128_return9318_len1000.pt \
  agent.otil.input_type=state_state \
  agent.otil.loss_best_of_k_k=8 \
  agent.otil.loss_ot_cost_type=l2 \
  agent.otil.loss_use_huber_speedup=false \
  agent.otil.loss_use_detached_prev_obs=false \
  agent.otil.critic_reward_mapping=exp \
  agent.otil.critic_reward_shapping=true
```

For another DFlex task, change `task.env.env_name` and the demonstration path.
Keep `agent.ppo.horizon_len` or `agent.sac.horizon_len` equal to the FOCUS
horizon. For SAC, `warm_up` must equal `horizon_len`; the supplied config ties
them together.
