"""Portable, commit-pinned Slurm exports for the filesystem supervisor."""

import hashlib
import json
import os
import re
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


def _git(*args):
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout.strip()


def managed_source(commit, remote=None, branch=None, require_pushed=False):
    """Git source for a checkout the supervisor clones per branch under its cluster root.

    The remote defaults to origin's URL and the branch to the current one. Jobs archive the
    pinned commit from that checkout, so the commit must be on the remote branch. An scp-style
    SSH origin (``git@host:owner/repo``) is recorded as HTTPS, which clones a public repository
    without credentials on the cluster; pass ``remote`` to keep SSH.
    """
    if remote is None:
        remote = _git('remote', 'get-url', 'origin')
        scp_style = re.fullmatch(r'git@([A-Za-z0-9.-]+):(.+)', remote)
        if scp_style:
            remote = f'https://{scp_style.group(1)}/{scp_style.group(2)}'
    if branch is None:
        branch = _git('rev-parse', '--abbrev-ref', 'HEAD')
        if branch == 'HEAD':
            raise ValueError('Detached HEAD: pass --git_branch, the branch the cluster checkout should track')
    pushed = None
    try:
        origin = _git('remote', 'get-url', 'origin')
        if remote in (origin, re.sub(r'^git@([A-Za-z0-9.-]+):', r'https://\1/', origin)):
            pushed = subprocess.run(
                ['git', 'merge-base', '--is-ancestor', commit, f'refs/remotes/origin/{branch}'], capture_output=True
            ).returncode == 0
    except subprocess.CalledProcessError:
        pass
    if pushed is False:
        message = f'commit {commit[:12]} is not on origin/{branch} (as last fetched): push it before the job runs'
        if require_pushed:
            raise ValueError(message)
        print('NOTE: ' + message)
    elif pushed is None:
        print(f'NOTE: cannot check locally that {commit[:12]} is on {remote} {branch}; the supervisor checks it')
    return {'remote': remote, 'branch': branch, 'commit': commit}


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


# Inputs under the supervisor's cluster root, <root>/inputs/<relative>; the root reaches
# jobs as MCP_RUNS_DIR. Stored with this prefix and expanded by the job's shell.
MCP_INPUTS = '${MCP_RUNS_DIR}/inputs/'


def mcp_inputs(values):
    inputs = {}
    for relative in values or []:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', relative) or any(
            part in ('.', '..') for part in relative.split('/')
        ):
            raise ValueError('--mcp_input must be a relative path such as experts/demos (no traversal)')
        inputs[relative] = MCP_INPUTS + relative
    return inputs


def shell_path(value, suffix=''):
    """Shell word for an input path: quoted, or expanded from MCP_RUNS_DIR for --mcp_input."""
    if value.startswith(MCP_INPUTS):
        # mcp_inputs() restricts the relative part to [A-Za-z0-9_./-], safe inside double quotes
        return '"${MCP_RUNS_DIR:?MCP_RUNS_DIR is set by the cluster supervisor}/inputs/' + value[len(MCP_INPUTS) :] + suffix + '"'
    return shlex.quote(value + suffix)


def run_identity():
    name = 'mineral-' + uuid.uuid4().hex
    return name, uuid.uuid5(uuid.NAMESPACE_URL, 'cluster-supervisor:' + name).hex


def bootstrap(repo, commit, run_id, inputs, docker, image=None):
    """Prepare source without touching the shared checkout or fetching moving refs.

    With an Apptainer ``image``, kernel caches persist in ``$MCP_RUNS_DIR/cache/warp/<image name>`` (bound to
    /cache/warp): the first job compiles Warp's kernels, later jobs on the same image reuse them.

    ``repo=None``: the supervisor prepares a per-branch checkout and passes its path
    in ``MCP_CHECKOUT`` (it clones and fetches; the job only reads the commit).
    """
    checkout = 'CLUSTER_REPO="${MCP_CHECKOUT:?MCP_CHECKOUT is set by the cluster supervisor}"' if repo is None else (
        f'CLUSTER_REPO={shlex.quote(repo)}'
    )
    lines = [
        'set -euo pipefail',
        'RUN_ROOT="$PWD"',
        f'export CLUSTER_RUN_ID={shlex.quote(run_id)}',
        f'export MINERAL_GIT_COMMIT={shlex.quote(commit)}',
        'export APPTAINERENV_CLUSTER_RUN_ID="$CLUSTER_RUN_ID"',
        'export APPTAINERENV_MINERAL_GIT_COMMIT="$MINERAL_GIT_COMMIT"',
        checkout,
        'git -C "$CLUSTER_REPO" cat-file -e "$MINERAL_GIT_COMMIT^{commit}"',
        'mkdir "$RUN_ROOT/source"',
        'git -C "$CLUSTER_REPO" archive "$MINERAL_GIT_COMMIT" | tar -x -C "$RUN_ROOT/source"',
        'mkdir -p "$RUN_ROOT/results"',
        'export MINERAL_RUN_RESULTS="$RUN_ROOT/results"',
    ]
    for relative, absolute in inputs.items():
        dest = '"$RUN_ROOT/source"/' + shlex.quote(relative)
        lines.append(f'test -e {shell_path(absolute)} || {{ echo "missing input: "{shell_path(absolute)} >&2; exit 1; }}')
        if not docker:
            lines.extend([
                f'mkdir -p "$(dirname {dest})"',
                f'ln -s {shell_path(absolute)} {dest}',
            ])
    if docker and image:
        # Warp builds straight into its cache folder, so concurrent jobs must not share one: each job compiles into
        # a private copy seeded from the shared cache, and on exit publishes new modules with an atomic rename.
        name = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(str(image)).stem) or 'image'
        lines.extend([
            f'WARP_CACHE_SHARED="${{MCP_RUNS_DIR:?MCP_RUNS_DIR is set by the cluster supervisor}}/cache/warp/{name}"',
            'mkdir -p "$WARP_CACHE_SHARED"',
            'WARP_CACHE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mineral-warp-cache.XXXXXX")"',  # node-local, not downloaded
            'cp -a "$WARP_CACHE_SHARED/." "$WARP_CACHE_ROOT/" 2>/dev/null || true',
            'publish_warp_cache() {',
            '  for module in "$WARP_CACHE_ROOT"/*/; do',
            '    target="$WARP_CACHE_SHARED/$(basename "$module")"',
            '    [ -e "$target" ] && continue',
            '    staging="$WARP_CACHE_SHARED/.staging-$$-$(basename "$module")"',
            '    cp -a "$module" "$staging" && mv -T "$staging" "$target" 2>/dev/null || rm -rf "$staging"',
            '  done',
            '  rm -rf "$WARP_CACHE_ROOT"',
            '}',
            'trap publish_warp_cache EXIT',
            'export APPTAINERENV_WARP_CACHE_PATH=/cache/warp',
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
