"""Exercise generated jobs without contacting a cluster or starting training."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mineral.common.cluster_jobs import cluster_inputs

SPAWNER = Path(__file__).resolve().parents[1] / 'spawner.py'


@pytest.fixture
def project(tmp_path):
    repo = tmp_path / 'cluster repo'
    repo.mkdir()
    (repo / 'version.txt').write_text('committed version')
    (repo / 'task.yaml').write_text('task:\n  name: DFlex\nseed: 110\nlabel: hello world\n')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'Test fixture'], check=True)
    return repo


def generate(project, state, *extra):
    return subprocess.run([
        sys.executable, str(SPAWNER), '--task_name', str(project / 'task.yaml'),
        '--env_files', 'task.yaml', '--deployment', 'slurm', '--device', 'cpu',
        '--supervisor_dir', str(state), '--cluster_repo', str(project), '--runtime', '1m',
        *extra,
    ], cwd=project, capture_output=True, text=True)


def shims(tmp_path):
    binary = tmp_path / 'bin'
    binary.mkdir()
    (binary / 'srun').write_text('#!/bin/bash\nif [[ "$1" == -n ]]; then shift 2; fi\nexec "$@"\n')
    (binary / 'uv').write_text(f'''#!{sys.executable}
import json,os,sys
from pathlib import Path
result={{'args':sys.argv[1:], 'version':Path('version.txt').read_text(),
        'run_id':os.environ['CLUSTER_RUN_ID'], 'commit':os.environ['MINERAL_GIT_COMMIT'],
        'slurm_id':os.environ['SLURM_JOB_ID']}}
Path(os.environ['MINERAL_RUN_RESULTS'],'captured.json').write_text(json.dumps(result))
''')
    for file in binary.iterdir():file.chmod(0o755)
    return dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'], SLURM_JOB_ID='5678')


def test_preview_executes_pinned_source_and_isolates_outputs(project, tmp_path):
    (project / 'version.txt').write_text('uncommitted change')
    state = tmp_path / 'state'
    generated = generate(project, state, '--priority', '70')
    assert generated.returncode == 0, generated.stderr
    assert 'uncommitted code changes are NOT included' in generated.stdout
    assert not (state / 'incoming').exists()
    run = next((state / 'previews').iterdir())
    script = (run / 'job.sbatch').read_text()
    assert '--gres=gpu' not in script
    subprocess.run(['bash', '-n', str(run / 'job.sbatch')], check=True)
    result = subprocess.run(['bash', str(run / 'job.sbatch')], cwd=run, env=shims(tmp_path), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    captured = json.loads((run / 'results/captured.json').read_text())
    metadata = json.loads((run / 'run.json').read_text())
    assert captured['version'] == 'committed version'
    assert captured['run_id'] == run.name == metadata['run_id']
    assert captured['slurm_id'] == '5678'
    assert captured['commit'] == metadata['git_commit']
    assert 'seed=110' in captured['args']
    assert 'label="hello world"' in captured['args']
    assert 'logdir=${oc.env:MINERAL_RUN_RESULTS}' in captured['args']
    assert (project / 'version.txt').read_text() == 'uncommitted change'
    assert not (project / 'results').exists()
    assert (run / 'out').is_dir()
    assert metadata['priority'] == 70


def test_apptainer_arguments_and_inputs_are_quoted(project, tmp_path):
    inputs = tmp_path / 'demonstration data'
    inputs.mkdir()
    state = tmp_path / 'state'
    generated = generate(project, state, '--docker', '--docker_image', '/images/mineral test.sif',
                         '--cluster_input', f'experts/demos={inputs}')
    assert generated.returncode == 0, generated.stderr
    run = next((state / 'previews').iterdir())
    env = shims(tmp_path)
    binary = tmp_path / 'bin/apptainer'
    binary.write_text(f'''#!{sys.executable}
import json,os,sys
from pathlib import Path
Path(os.environ['MINERAL_RUN_RESULTS'],'container.json').write_text(json.dumps({{
'args':sys.argv[1:], 'run_id':os.environ['APPTAINERENV_CLUSTER_RUN_ID']}}))
''')
    binary.chmod(0o755)
    result = subprocess.run(['bash', str(run / 'job.sbatch')], cwd=run, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    captured = json.loads((run / 'results/container.json').read_text())
    assert captured['args'][0] == 'exec'
    assert f'{run}/source:/workspace' in captured['args']
    assert f'{run}/results:/results' in captured['args']
    assert f'{inputs}:/workspace/experts/demos:ro' in captured['args']
    assert '/images/mineral test.sif' in captured['args']
    assert captured['run_id'] == run.name
    assert 'logdir=/results' in captured['args'][-1]


def test_enqueue_publishes_matching_ready_manifest(project, tmp_path):
    state = tmp_path / 'state'
    result = generate(project, state, '--enqueue')
    assert result.returncode == 0, result.stderr
    manifest = next((state / 'incoming').glob('*.ready.json'))
    name = json.loads(manifest.read_text())['name']
    assert manifest.name == name+'.ready.json'
    assert (state / 'incoming' / name / 'job.sbatch').is_file()
    assert (state / 'incoming' / name / 'config.json').is_file()
    assert (state / 'incoming' / name / 'out').is_dir()


def test_task_cannot_enable_direct_submission(project, tmp_path):
    (project / 'task.yaml').write_text('seed: 1\nspawner:\n  deploy_now: true\n')
    result = generate(project, tmp_path / 'state')
    assert result.returncode != 0
    assert 'deploy_now must be supplied on the command line' in result.stderr


@pytest.mark.parametrize('mapping', ['../escape=/data', '/absolute=/data', 'data=relative', 'data=/bad:path'])
def test_invalid_input_mapping(mapping):
    with pytest.raises(ValueError):cluster_inputs([mapping])
