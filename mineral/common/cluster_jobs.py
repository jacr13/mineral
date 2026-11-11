"""Portable, commit-pinned Slurm exports for the filesystem supervisor."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tempfile
import uuid


def source_revision(revision):
    result = subprocess.run(
        ['git', 'rev-parse', '--verify', f'{revision}^{{commit}}'],
        check=True, capture_output=True, text=True,
    )
    commit = result.stdout.strip()
    dirty = bool(subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        check=True, capture_output=True, text=True,
    ).stdout.strip())
    return commit, dirty


def cluster_inputs(values):
    inputs = {}
    for value in values or []:
        relative, separator, absolute = value.partition('=')
        path = PurePosixPath(relative)
        if (not separator or not relative or path.is_absolute() or
                any(part in ('', '.', '..') for part in relative.split('/')) or
                not absolute.startswith('/') or any(c in value for c in '\n\r,:')):
            raise ValueError('--cluster_input must be relative/path=/absolute/cluster/path (no traversal, commas or colons)')
        inputs[relative] = absolute
    return inputs


def run_identity():
    name = 'mineral-' + uuid.uuid4().hex
    return name, uuid.uuid5(uuid.NAMESPACE_URL, 'cluster-supervisor:' + name).hex


def bootstrap(repo, commit, run_id, inputs, docker):
    """Prepare source without touching the shared checkout or fetching moving refs."""
    lines = [
        'set -euo pipefail',
        'RUN_ROOT="$PWD"',
        f'export CLUSTER_RUN_ID={shlex.quote(run_id)}',
        f'export MINERAL_GIT_COMMIT={shlex.quote(commit)}',
        'export APPTAINERENV_CLUSTER_RUN_ID="$CLUSTER_RUN_ID"',
        'export APPTAINERENV_MINERAL_GIT_COMMIT="$MINERAL_GIT_COMMIT"',
        f'CLUSTER_REPO={shlex.quote(repo)}',
        'git -C "$CLUSTER_REPO" cat-file -e "$MINERAL_GIT_COMMIT^{commit}"',
        'mkdir "$RUN_ROOT/source"',
        'git -C "$CLUSTER_REPO" archive "$MINERAL_GIT_COMMIT" | tar -x -C "$RUN_ROOT/source"',
        'mkdir -p "$RUN_ROOT/results"',
        'export MINERAL_RUN_RESULTS="$RUN_ROOT/results"',
    ]
    for relative, absolute in inputs.items():
        dest = '"$RUN_ROOT/source"/' + shlex.quote(relative)
        lines.append(f'test -e {shlex.quote(absolute)}')
        if not docker:
            lines.extend([
                f'mkdir -p "$(dirname {dest})"',
                f'ln -s {shlex.quote(absolute)} {dest}',
            ])
    lines.append('cd "$RUN_ROOT/source"')
    return '\n'.join(lines)


def write_export(state_dir, name, run_id, script, config, metadata, enqueue=False):
    """Previews are inert. Enqueue publishes a complete directory before its manifest."""
    root = Path(state_dir).expanduser().resolve()
    parent = root / ('incoming' if enqueue else 'previews')
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / (name if enqueue else run_id)
    temporary = Path(tempfile.mkdtemp(prefix='.tmp-', dir=parent))
    try:
        (temporary / 'out').mkdir()  # Slurm opens these files before executing bash.
        (temporary / 'job.sbatch').write_text(script)
        (temporary / 'job.sbatch').chmod(0o755)
        config_text = json.dumps(config, indent=2) + '\n'
        (temporary / 'config.json').write_text(config_text)
        metadata = dict(metadata, schema_version=1, name=name, run_id=run_id,
                        config_sha256=hashlib.sha256(config_text.encode()).hexdigest(),
                        script_sha256=hashlib.sha256(script.encode()).hexdigest())
        (temporary / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
        # Flush payloads before publishing the ready marker.
        for path in temporary.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:os.fsync(stream.fileno())
        temporary.rename(target)
        if enqueue:
            fd, manifest = tempfile.mkstemp(prefix='.manifest-', dir=parent)
            with os.fdopen(fd, 'w') as stream:
                json.dump({'name': name}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(manifest, parent / f'{name}.ready.json')
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return target
