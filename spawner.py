import argparse
import os
import random
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import yaml

CALIBERS = [
    {
        "time": "0-00:10:00",
        "partition": {"cpu": "debug-cpu", "gpu": "shared-gpu"},
    },
    {
        "time": "0-12:00:00",
        "partition": {"cpu": "shared-cpu", "gpu": "shared-gpu"},
    },
    {
        "time": "4-00:00:00",
        "partition": {
            "cpu": "public-cpu,private-cui-cpu",
            "gpu": "private-cui-gpu,private-kalousis-gpu",
        },
    },
    {
        "time": "7-00:00:00",
        "partition": {
            "cpu": "public-longrun-cpu,private-cui-cpu",
            "gpu": "private-cui-gpu,private-kalousis-gpu",
        },
    },
]

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

LOCAL_SCRIPT_CONTENT = """#!/usr/bin/env bash

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


def _parse_cli_overrides(raw_overrides):
    overrides = []
    for raw in raw_overrides or []:
        if "=" not in raw:
            raise ValueError(f"Override '{raw}' must be in the form key=value.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override '{raw}' has an empty key.")
        parsed_value = yaml.safe_load(value)
        overrides.append((key, parsed_value))
    return overrides


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
        for key, value in zip(keys, combination, strict=False):
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


def _write_local_script(script_path, name, command, args):
    if args.docker:
        if args.docker_image is None:
            raise ValueError("Docker image must be specified when using Docker.")
        command = DOCKER_CMD.format(
            docker_image=args.docker_image,
            command=command,
        )

    script_path.write_text(LOCAL_SCRIPT_CONTENT.format(name=name, command=command))
    script_path.chmod(0o755)


def _write_slurm_script(script_path, name, command, args):
    def _get_partition_and_duration(runtime, device):
        """Given a runtime string (e.g. '12h', '30m', '1d', or '0-12:00:00'),
        return the most suitable partition configuration.
        """

        # --- Parse runtime string into timedelta ---
        def to_timedelta(s: str) -> timedelta:
            if "-" in s:  # format like 0-12:00:00
                days, hms = s.split("-")
                hours, minutes, seconds = map(int, hms.split(":"))
                return timedelta(days=int(days), hours=hours, minutes=minutes, seconds=seconds)
            else:
                t = int(s[:-1])
                if s.endswith("s"):
                    return timedelta(seconds=t), f"0-00:00:{t:02d}"
                elif s.endswith("m"):
                    return timedelta(minutes=t), f"0-00:{t:02d}:00"
                elif s.endswith("h"):
                    return timedelta(hours=t), f"0-{t:02d}:00:00"
                elif s.endswith("d"):
                    return timedelta(days=t), f"{t}-00:00:00"
                else:
                    raise ValueError(f"Invalid runtime format: {s}")

        input_time, formatted_duration = to_timedelta(runtime)

        # --- Select best partition ---
        best_partition = CALIBERS[0]["partition"][device]
        for rule in CALIBERS:
            if to_timedelta(rule["time"]) <= input_time:
                best_partition = rule["partition"][device]
            else:
                break

        return best_partition, formatted_duration

    partition, duration = _get_partition_and_duration(args.runtime, args.device)
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
    base_tasks_root = Path("tasks").resolve()

    if args.task_name:
        task_path = Path(args.task_name)
        if task_path.exists():
            task_path = task_path.resolve()
        else:
            candidate = base_tasks_root / task_path
            if candidate.exists():
                task_path = candidate.resolve()
            else:
                raise FileNotFoundError(f"Task entry '{args.task_name}' not found (checked '{task_path}' and '{candidate}').")
    else:
        task_path = base_tasks_root

    if not task_path.exists():
        raise FileNotFoundError(f"No task definitions in '{task_path}'.")

    if task_path.is_dir():
        configs = [path.resolve() for path in sorted(task_path.rglob("*.yaml"))]
        if not configs:
            raise ValueError(f"No YAML task files found under '{task_path}'.")
    elif task_path.is_file():
        if task_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Task file '{task_path}' must be a YAML file.")
        configs = [task_path.resolve()]
    else:
        raise ValueError(f"Unsupported task entry type: '{task_path}'.")

    spawn_root = Path("spawn")
    spawn_root.mkdir(parents=True, exist_ok=True)

    created_scripts = []
    cli_overrides = _parse_cli_overrides(args.overrides)

    for config_path in configs:
        try:
            rel_path = config_path.relative_to(base_tasks_root)
        except ValueError:
            rel_path = Path(config_path.name)
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
                (f"{base_name}_sweep{index:03d}", variant_config) for index, variant_config in enumerate(sweep_variants)
            ]
            if not variant_entries:
                continue
        else:
            variant_entries = [(base_name, deepcopy(task_config))]

        for sweep_index, (variant_name, variant_config) in enumerate(variant_entries, start=1):
            timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_component = ""
            duplicate_component = ""

            if args.timestamp:
                timestamp_component = f"_{timestamp_base}"
                script_basename = f"{variant_name}{timestamp_component}"
                script_path = job_subdir / f"{script_basename}.sh"
                counter = 1
                while script_path.exists():
                    duplicate_component = f"_{counter:02d}"
                    script_basename = f"{variant_name}{timestamp_component}{duplicate_component}"
                    script_path = job_subdir / f"{script_basename}.sh"
                    counter += 1
            else:
                script_basename = variant_name
                script_path = job_subdir / f"{script_basename}.sh"
                counter = 1
                while script_path.exists():
                    duplicate_component = f"_{counter:02d}"
                    script_basename = f"{variant_name}{duplicate_component}"
                    script_path = job_subdir / f"{script_basename}.sh"
                    counter += 1
                timestamp_component = f"_{timestamp_base}"

            job_name = script_basename

            if args.sweep and args.sweep_logdir:
                base_logdir = variant_config.get("logdir")
                experiment_name = None
                if isinstance(base_logdir, str):
                    stripped = base_logdir.strip('"')
                    if stripped.startswith("workdir/"):
                        parts = stripped.split("/")
                        if len(parts) > 1 and parts[1]:
                            experiment_name = parts[1]
                if not experiment_name:
                    experiment_name = variant_name

                logdir_suffix = timestamp_component or f"_{timestamp_base}"
                if duplicate_component:
                    logdir_suffix = f"{logdir_suffix}{duplicate_component}"
                experiment_with_time = f"{experiment_name}{logdir_suffix}"
                variant_config["logdir"] = f"workdir/{experiment_with_time}/sweep_{sweep_index}"

            effective_config = deepcopy(variant_config)
            for key, value in cli_overrides:
                _set_nested_value(effective_config, key.split("."), value)

            overrides = _build_overrides(effective_config)
            command = _command_from_overrides(overrides)

            if args.deployment == "slurm":
                _write_slurm_script(script_path, job_name, command, args)
                if args.deploy_now:
                    subprocess.run(["sbatch", str(script_path)], check=True)
            else:
                _write_local_script(script_path, job_name, command, args)
                if args.deploy_now:
                    subprocess.run(["bash", str(script_path)], check=True)

            created_scripts.append(script_path)
            if args.cleanup:
                script_path.unlink()

    if not args.deploy_now or not args.cleanup:
        for script_path in created_scripts:
            print(f"Created task script: {script_path}")


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Job Spawner")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument(
        "--deployment",
        type=str,
        choices=["local", "slurm"],
        default="local",
        help="deployment backend",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Which device?",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default="12h",
        help="job runtime use format d-hh:mm:ss, or number of minutes, hours, days e.g., 30m, 2h, 1d",
    )
    boolean_flag(parser, "deploy_now", default=False, help="deploy immediately?")
    boolean_flag(parser, "sweep", default=False, help="hp search?")
    boolean_flag(parser, "cleanup", default=False, help="remove script files after deployment")
    boolean_flag(parser, "timestamp", default=True, help="append timestamp (YYYYMMDD_HHMMSS) to generated script filenames")
    boolean_flag(parser, "sweep_logdir", default=True, help="override logdir per sweep with timestamped subdirectories")
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
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help=(
            "Override config values (can be repeated). "
            "Use dot paths, e.g. --set wandb.mode=offline --set agent.otil.max_epochs=6000."
        ),
    )
    boolean_flag(parser, "docker", default=False, help="use docker?")
    parser.add_argument(
        "--docker_image",
        type=str,
        default=None,
        help="Name of docker image or path to image (cluster)",
    )

    # args not used yet:
    parser.add_argument("--env_bundle", type=str, default=None)
    parser.add_argument("--demo_dir", type=str, default=None)
    parser.add_argument("--num_demos", "--list", nargs="+", type=str, default=None)

    args = parser.parse_args()

    # Create (and optionally deploy) the jobs
    run(args)
