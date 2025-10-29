import argparse
import subprocess
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
        return [f"{prefix}={_format_list(value)}"]
    return [f"{prefix}={_format_scalar(value)}"]


def _build_overrides(config):
    overrides = []
    for key, value in config.items():
        if isinstance(value, dict) and "name" in value:
            overrides.append(f"{key}={_format_scalar(value['name'])}")
            nested = {k: v for k, v in value.items() if k != "name"}
            for nested_key, nested_value in nested.items():
                overrides.extend(
                    _flatten_value(f"{key}.{nested_key}", nested_value),
                )
        else:
            overrides.extend(_flatten_value(key, value))
    return overrides


def _command_from_overrides(overrides):
    if not overrides:
        return "python -m mineral.scripts.run"
    lines = ["python -m mineral.scripts.run \\"]
    for index, override in enumerate(overrides):
        suffix = " \\" if index < len(overrides) - 1 else ""
        lines.append(f"{override}{suffix}")
    return "\n".join(lines)


def _write_tmux_script(script_path, name, command):
    script_path.write_text(TMUX_FILE_CONTENT.format(name=name, command=command))
    script_path.chmod(0o755)


def _write_slurm_script(script_path, name, command, caliber):
    partition = caliber["partition"].get("gpu", caliber["partition"]["cpu"])
    duration = caliber["time"]
    script_content = SBATCH_FILE_CONTENT.format(
        name=name,
        partition=partition,
        num_workers=1,
        duration=duration,
        memory=32,
        extra_params=SBATCH_GPU,
        modules="",
        command=command,
    )
    script_path.write_text(script_content)
    script_path.chmod(0o755)


def run(args):
    tasks_root = Path("tasks")
    if args.prefix:
        tasks_root = tasks_root / args.prefix

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
        job_name = rel_path.with_suffix("").name
        job_subdir = spawn_root / rel_path.parent
        job_subdir.mkdir(parents=True, exist_ok=True)
        script_path = job_subdir / f"{job_name}.sh"

        with config_path.open("r", encoding="utf-8") as file:
            task_config = yaml.safe_load(file) or {}

        overrides = _build_overrides(task_config)
        command = _command_from_overrides(overrides)

        if args.deployment == "slurm":
            caliber = CALIBERS[args.caliber]
            _write_slurm_script(script_path, job_name, command, caliber)
        else:
            _write_tmux_script(script_path, job_name, command)

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
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--env_bundle", type=str, default=None)
    parser.add_argument("--demo_dir", type=str, default=None)
    parser.add_argument("--num_trials", type=int, default=0)
    parser.add_argument(
        "--deployment",
        type=str,
        choices=["tmux", "slurm"],
        default="tmux",
        help="deploy how?",
    )
    parser.add_argument("--caliber", type=str, default="short", choices=CALIBERS.keys())
    boolean_flag(parser, "deploy_now", default=False, help="deploy immediately?")
    boolean_flag(parser, "sweep", default=False, help="hp search?")
    boolean_flag(parser, "clear", default=False, help="clear files after deployment")
    boolean_flag(parser, "wandb_upgrade", default=True, help="upgrade wandb?")
    parser.add_argument("--num_demos", "--list", nargs="+", type=str, default=None)
    parser.add_argument(
        "--wandb_base_url",
        type=str,
        default="https://api.wandb.ai",
        help="your wandb base url",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="your wandb api key",
    )
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
