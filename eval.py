"""Utilities for inspecting the `workdir` directory and replaying runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def _latest_run_dir(exp_root: Path) -> Path:
    """Return the most recent run directory inside ``exp_root``."""
    candidates = [p for p in exp_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {exp_root}")
    return max(candidates, key=lambda p: p.name)


def run_eval(exp_root: Path, *, render: bool = True) -> None:
    if not exp_root.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_root}")
    if not exp_root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {exp_root}")

    run_dir = _latest_run_dir(exp_root)
    ckpt_path = run_dir / "ckpt" / "final.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing final checkpoint: {ckpt_path}")

    resolved_config_path = run_dir / "resolved_config.yaml"
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Missing resolved_config.yaml: {resolved_config_path}")

    config = OmegaConf.load(resolved_config_path)
    OmegaConf.set_struct(config, False)

    config.run = "eval"
    config.ckpt = str(ckpt_path)
    config.ckpt_keys = config.get("ckpt_keys", "")
    config.logdir = str(run_dir).replace("workdir", "eval")

    if "wandb" in config:
        config.wandb.mode = "disabled"

    if "task" in config and "env" in config.task:
        config.task.env.render = render

    from mineral.scripts.run import main as mineral_run

    mineral_run(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and replay workdir runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "workdir",
        help="Directory to inspect (defaults to the repo's workdir).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Run evaluation with rendering.",
    )
    args = parser.parse_args()
    root = args.root

    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")
    if not root.is_dir():
        raise SystemExit(f"Path is not a directory: {root}")

    for child in root.iterdir():
        if child.is_dir():
            # if "SNU" not in child.name:
            #     continue
            print(child)
            try:
                run_eval(child, render=args.render)
            except Exception as e:
                print(f"Skipping {child}, because of {e}")
            # stop


if __name__ == "__main__":
    main()

    # add offset to render check rewarped
