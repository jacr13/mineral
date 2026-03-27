poetry update

run_dir="$(realpath plotter/workdir/DFlexHopper10M-OTIL_SHAC_20260121_011014/sweep_15)"
ckpt="$run_dir/ckpt/epochs100_steps204k_rewards869.14.pth"

python - <<PY
from omegaconf import OmegaConf

from mineral.scripts.run import main

run_dir = "${run_dir}"
ckpt = "${ckpt}"

cfg = OmegaConf.load(f"{run_dir}/resolved_config.yaml")
OmegaConf.set_struct(cfg, False)

#  -------- eval --------
# cfg.run = "eval"
# cfg.ckpt = ckpt
# cfg.wandb.mode = "disabled"
# cfg.task.env.render = True


#  -------- train --------

cfg.wandb.project = "OTIL_SAPO-dflex_hopper-debug-rebutal"
cfg.logdir = "workdir_hmap/DFlexHopper10M-OTIL_SHAC_ot_l2_Blues"
cfg.agent.shac.max_agent_steps=1500000

# not used
cfg.agent.otil.save_plan_heatmaps: True
cfg.agent.otil.save_plan_heatmaps_every: 1
cfg.agent.otil.save_plan_heatmaps_num_samples: 1

main(cfg)
PY

run_dir="$(realpath plotter/workdir/DFlexHopper10M-OTIL_SHAC_20260121_011014_01/sweep_19)"
ckpt="$run_dir/ckpt/epochs100_steps204k_rewards606.52.pth"

python - <<PY
from omegaconf import OmegaConf

from mineral.scripts.run import main

run_dir = "${run_dir}"
ckpt = "${ckpt}"

cfg = OmegaConf.load(f"{run_dir}/resolved_config.yaml")
OmegaConf.set_struct(cfg, False)

#  -------- eval --------
# cfg.run = "eval"
# cfg.ckpt = ckpt
# cfg.wandb.mode = "disabled"
# cfg.task.env.render = True


#  -------- train --------

cfg.wandb.project = "OTIL_SAPO-dflex_hopper-debug-rebutal"
cfg.logdir = "workdir_hmap/DFlexHopper10M-OTIL_SHAC_ot_cos_Blues"
cfg.agent.shac.max_agent_steps=1500000

# not used
cfg.agent.otil.save_plan_heatmaps: True
cfg.agent.otil.save_plan_heatmaps_every: 1
cfg.agent.otil.save_plan_heatmaps_num_samples: 1

main(cfg)
PY

