import json
import os
from pathlib import Path

import yaml


def load_yaml(path):
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing YAML file: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path):
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_yaml(path, data):
    yaml_path = Path(path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)


def sync_exp_from_remote(remote, remote_exp_dir, local_exp_dir):
    os.system(f"rsync -rvaP --progress {remote}:{remote_exp_dir} {local_exp_dir}")
