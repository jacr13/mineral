import argparse
import os
import random
import subprocess
from copy import deepcopy
from itertools import product
from pathlib import Path

import yaml

CALIBERS = {
    "debug": {
        "time": "0-00:10:00",
        "partition": {"cpu": "debug-cpu", "gpu": "shared-gpu"},
    },
    "veryveryshort": {
        "time": "0-00:30:00",
        "partition": {"cpu": "shared-cpu", "gpu": "shared-gpu"},
    },
    "veryshort": {
        "time": "0-03:00:00",
        "partition": {"cpu": "shared-cpu", "gpu": "shared-gpu"},
    },
    "short": {
        "time": "0-06:00:00",
        "partition": {"cpu": "shared-cpu", "gpu": "shared-gpu"},
    },
    "long": {
        "time": "0-12:00:00",
        "partition": {"cpu": "shared-cpu", "gpu": "shared-gpu"},
    },
    "daylong": {
        "time": "1-00:00:00",
        "partition": {
            "cpu": "public-cpu,private-cui-cpu",
            "gpu": "private-cui-gpu,private-kalousis-gpu",
        },
    },
    "verylong": {
        "time": "2-00:00:00",
        "partition": {
            "cpu": "public-cpu,private-cui-cpu",
            "gpu": "private-cui-gpu,private-kalousis-gpu",
        },
    },
    "veryverylong": {
        "time": "4-00:00:00",
        "partition": {
            "cpu": "public-cpu,private-cui-cpu",
            "gpu": "private-cui-gpu,private-kalousis-gpu",
        },
    },
    "veryveryverylong": {
        "time": "7-00:00:00",
        "partition": {
            "cpu": "public-longrun-cpu,private-cui-cpu",
            "gpu": "private-cui-gpu,private-kalousis-gpu",
        },
    },
}

SBATCH_GPU = "#SBATCH --gres=gpu:1"

SINGULARITY_CMD = """-n {num_workers} singularity run {cuda} -B $HOME/scratch:/scratch {sing_image} \\
    bash -c "cd {path2code}; \\
    {command}"
"""

SBATCH_FILE_CONTENT = """#!/usr/bin/env bash

#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --ntasks={num_workers}
#SBATCH --time={duration}
#SBATCH --mem={memory}000
#SBATCH --output=./out/run_%j.out
#SBATCH --error=./out/run_e%j.out
{extra_params}

{modules}

srun {command}
"""

DOCKER_CMD = """docker run \\
    -v $(pwd):/workspace \\
    {docker_image} {command}
"""

TMUX_FILE_CONTENT = """#!/usr/bin/env bash

# job name: {name}

{command}
"""

def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Args:
        parser (argparse.Parser): Parser to add the flag to
        name (str): Name of the flag
          --<name> will enable the flag, while --no-<name> will disable it
        default (bool or None): Default value of the flag
        help (str): Help string for the flag
    """
    dest = name.replace("-", "_")
    parser.add_argument(
        "--" + name,
        action="store_true",
        default=default,
        dest=dest,
        help=help,
    )
    parser.add_argument("--no-" + name, action="store_false", dest=dest)

def get_gitsha():
    _gitsha = "gitSHA_{}"
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        gitsha = _gitsha.format(out.strip().decode("ascii"))
    except OSError:
        gitsha = "noGitSHA"
    return gitsha


def _format_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(value, "g")
    if value is None:
        return "null"
    value_str = str(value)
    if not value_str:
        return '""'
    if "$(" in value_str:
        return f'"{value_str}"'
    if any(char.isspace() for char in value_str):
        return f'"{value_str}"'
    return value_str


def _format_list(values):
    formatted = []
    for item in values:
        if isinstance(item, list):
            formatted.append(_format_list(item))
        elif isinstance(item, dict):
            raise ValueError("Nested dictionaries inside lists are not supported.")
        else:
            formatted.append(_format_scalar(item))
    return f"[{','.join(formatted)}]"


def _flatten_value(prefix, value):
    if isinstance(value, dict):
        overrides = []
        for key, sub_value in value.items():
            overrides.extend(_flatten_value(f"{prefix}.{key}", sub_value))
        return overrides
    if isinstance(value, list):
        return [f"\t\t{prefix}={_format_list(value)}"]
    return [f"\t\t{prefix}={_format_scalar(value)}"]


def _build_overrides(config):
    overrides = []
    for key, value in config.items():
        if isinstance(value, dict) and "name" in value:
            overrides.append(f"\t\t{key}={_format_scalar(value['name'])}")
            nested = {k: v for k, v in value.items() if k != "name"}
            for nested_key, nested_value in nested.items():
                overrides.extend(
                    _flatten_value(f"{key}.{nested_key}", nested_value),
                )
        else:
            overrides.extend(_flatten_value(key, value))
    return overrides


def _set_nested_value(config, path, value):
    cursor = config
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value


def _generate_sweep_configs(base_config, sweep_spec, mode, max_variants):
    keys = list(sweep_spec.keys())
    options = []
    for key in keys:
        values = sweep_spec[key]
        if not isinstance(values, list):
            raise ValueError(
                f"Sweep entry '{key}' must be a list, got '{type(values).__name__}'.",
            )
        options.append(values)

    combinations = list(product(*options))
    if not combinations:
        return []

    if mode == "random":
        random.shuffle(combinations)

    limit = None
    # sweep_max == 0 means "no limit" so only apply the cap when positive.
    if max_variants is not None and max_variants > 0:
        limit = min(max_variants, len(combinations))

    if limit is not None:
        combinations = combinations[:limit]

    variants = []
    for combination in combinations:
        config_variant = deepcopy(base_config)
        for key, value in zip(keys, combination):
            _set_nested_value(config_variant, key.split("."), value)
        variants.append(config_variant)
    return variants


def _command_from_overrides(overrides):
    if not overrides:
        return "python -m mineral.scripts.run"
    lines = ["python -m mineral.scripts.run \\"]
    for index, override in enumerate(overrides):
        suffix = " \\" if index < len(overrides) - 1 else ""
        lines.append(f"{override}{suffix}")
    return "\n".join(lines)


def _write_tmux_script(script_path, name, command, args):
    if args.docker:
        if args.docker_image is None:
            raise ValueError("Docker image must be specified when using Docker.")
        command = DOCKER_CMD.format(
            docker_image=args.docker_image,
            command=command,
        )

    script_path.write_text(TMUX_FILE_CONTENT.format(name=name, command=command))
    script_path.chmod(0o755)

def _write_slurm_script(script_path, name, command, args):
    caliber = CALIBERS[args.caliber]
    partition = caliber["partition"][args.device]
    duration = caliber["time"]

    num_workers = 1
    memory = 32

    if args.docker:
        if args.docker_image is None:
            raise ValueError("Docker image must be specified when using Singularity.")
        command = SINGULARITY_CMD.format(
            num_workers=num_workers,
            cuda="--nv" if args.device == "gpu" else "",
            sing_image=args.docker_image,
            path2code=os.getcwdb().decode(),
            command=command,
        )
    script_content = SBATCH_FILE_CONTENT.format(
        name=name,
        partition=partition,
        num_workers=num_workers,
        duration=duration,
        memory=memory,
        extra_params=SBATCH_GPU,
        modules="",
        command=command,
    )
    script_path.write_text(script_content)
    script_path.chmod(0o755)


def run(args):
    tasks_root = Path("tasks")
    if args.task_name:
        tasks_root = tasks_root / args.task_name

    if not tasks_root.exists():
        raise FileNotFoundError(f"No task definitions in '{tasks_root}'.")

    spawn_root = Path("spawn")
    spawn_root.mkdir(parents=True, exist_ok=True)

    configs = sorted(tasks_root.rglob("*.yaml"))
    if not configs:
        raise ValueError(f"No YAML task files found under '{tasks_root}'.")

    created_scripts = []
    for config_path in configs:
        rel_path = config_path.relative_to(Path("tasks"))
        job_subdir = spawn_root / rel_path.parent
        job_subdir.mkdir(parents=True, exist_ok=True)

        with config_path.open("r", encoding="utf-8") as file:
            task_config = yaml.safe_load(file) or {}

        sweep_spec = task_config.pop("sweep", None)
        base_name = rel_path.with_suffix("").name

        if args.sweep:
            if not sweep_spec:
                continue  # Skip configs without sweep definitions when sweep mode is requested
            sweep_variants = _generate_sweep_configs(
                task_config,
                sweep_spec,
                args.sweep_mode,
                args.sweep_max,
            )
            variant_entries = [
                (f"{base_name}_sweep{index:03d}", variant_config)
                for index, variant_config in enumerate(sweep_variants)
            ]
            if not variant_entries:
                continue
        else:
            variant_entries = [(base_name, task_config)]

        for variant_name, variant_config in variant_entries:
            script_path = job_subdir / f"{variant_name}.sh"

            overrides = _build_overrides(variant_config)
            command = _command_from_overrides(overrides)

            if args.deployment == "slurm":
                _write_slurm_script(script_path, variant_name, command, args)
            else:
                _write_tmux_script(script_path, variant_name, command, args)

            created_scripts.append(script_path)

            if args.deploy_now:
                subprocess.run(["bash", str(script_path)], check=True)
                if args.clear:
                    script_path.unlink()

    if not args.deploy_now or not args.clear:
        for script_path in created_scripts:
            print(f"Created task script: {script_path}")



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Job Spawner")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--env_bundle", type=str, default=None)
    parser.add_argument("--demo_dir", type=str, default=None)
    parser.add_argument(
        "--deployment",
        type=str,
        choices=["tmux", "slurm"],
        default="tmux",
        help="deploy how?",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Which device?",
    )
    parser.add_argument("--caliber", type=str, default="short", choices=CALIBERS.keys())
    boolean_flag(parser, "deploy_now", default=False, help="deploy immediately?")
    boolean_flag(parser, "sweep", default=False, help="hp search?")
    boolean_flag(parser, "clear", default=False, help="clear files after deployment")
    parser.add_argument(
        "--sweep_max",
        type=int,
        default=5,
        help="Upper bound on generated sweep variants; set to 0 for no cap.",
    )
    parser.add_argument(
        "--sweep_mode",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Variant selection strategy: full grid or random sampling (before sweep_max).",
    )
    parser.add_argument("--num_demos", "--list", nargs="+", type=str, default=None)
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="team or personal username",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="wandb project name",
    )
    boolean_flag(parser, "docker", default=False, help="use docker?")
    parser.add_argument(
        "--docker_image",
        type=str,
        default=None,
        help="Name of docker image or path to image (cluster)",
    )
    args = parser.parse_args()

    # Create (and optionally deploy) the jobs
    run(args)
