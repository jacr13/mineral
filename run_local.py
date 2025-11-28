#!/usr/bin/env python

"""Utility to launch locally generated experiment scripts under the spawn/ tree."""

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List


def _resolve_target(spawn_root: Path, target: Path) -> Path:
    """Resolve a user-provided target to an existing path."""
    if target.is_absolute():
        return target

    if target.exists():
        return target.resolve()

    candidate = (spawn_root / target).resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Target '{target}' does not exist (checked '{target}' and '{candidate}').")


def _collect_scripts(spawn_root: Path, targets: Iterable[Path]) -> List[Path]:
    """Collect all .sh scripts from the provided targets."""
    scripts = []

    for raw_target in targets:
        resolved = _resolve_target(spawn_root, raw_target)
        if resolved.is_file():
            if resolved.suffix != ".sh":
                raise ValueError(f"File '{resolved}' is not a .sh script.")
            scripts.append(resolved)
            continue

        if resolved.is_dir():
            scripts.extend(sorted(path for path in resolved.rglob("*.sh") if path.is_file()))
            continue

        raise ValueError(f"Unsupported target type: {resolved}")

    deduped = []
    seen = set()
    for script in scripts:
        if script not in seen:
            deduped.append(script)
            seen.add(script)
    return deduped


def _run_scripts(scripts: List[Path], dry_run: bool, keep_going: bool) -> None:
    for index, script in enumerate(scripts, start=1):
        print(f"[{index}/{len(scripts)}] Launching {script}")
        if dry_run:
            continue
        try:
            subprocess.run(["bash", str(script)], check=True)
        except subprocess.CalledProcessError as err:
            print(f"Script failed: {script} (exit code {err.returncode})")
            if not keep_going:
                raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generated local experiment scripts from the spawn directory.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        help="Script paths or directories to execute. Relative paths are resolved under spawn/ by default.",
    )
    parser.add_argument(
        "--spawn-root",
        type=Path,
        default=Path("spawn"),
        help="Root directory containing generated scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List scripts without executing them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running remaining scripts even if one fails.",
    )

    args = parser.parse_args()

    spawn_root = args.spawn_root.resolve()
    if not spawn_root.exists():
        raise SystemExit(f"Spawn root does not exist: {spawn_root}")
    if not args.targets:
        targets = [spawn_root]
    else:
        targets = args.targets

    scripts = _collect_scripts(spawn_root, targets)
    if not scripts:
        raise SystemExit(f"No .sh scripts found under: {', '.join(map(str, targets))}")

    _run_scripts(scripts, args.dry_run, args.keep_going)


if __name__ == "__main__":
    main()
