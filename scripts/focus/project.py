"""Which paper's experiments the launch and table tools serve (one pipeline, two projects).

The tools live in scripts/focus/ of every checkout of the shared repository; the project is the folder that holds the
checkout (``<project>/code``): ``.../focus/code`` → FOCUS, ``.../mentor/code`` → MENTOR. Override with the
environment variable MINERAL_PROJECT=focus|mentor. Each project keeps its own manifest and tables
(``<project>/results/data``, ``<project>/results/tables``), run registry, generated task folder, branch and run tag;
seeds (scripts/focus/seeds.yaml), the table rules and the normalisation are shared.
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))  # .../<project>
CODE = os.path.join(ROOT, 'code')

PROFILES = {
    'focus': dict(
        scripts=HERE,  # design.yaml and make_task_files.py
        task_dir='tasks/focus',  # generated task files: <task_dir>/<batch>/<task>_<arm>.yaml
        task_name='focus',  # spawner --task_name (relative to tasks/)
        branch='focus-wip',
        tag='focus',  # first word of the run registry line
        run_index=os.path.join(ROOT, 'agent/run_index.tsv'),
    ),
    'mentor': dict(
        scripts=os.path.join(CODE, 'scripts/mentor'),
        task_dir='tasks/mentor/icml2',
        task_name='mentor/icml2',
        branch='mentor-wip',
        tag='icml2',
        run_index=os.path.join(ROOT, 'wip/run_index.tsv'),
    ),
}

NAME = os.environ.get('MINERAL_PROJECT') or os.path.basename(ROOT)
if NAME not in PROFILES:
    raise SystemExit(f'unknown project {NAME!r} (folder {ROOT}); set MINERAL_PROJECT to one of {sorted(PROFILES)}')
P = PROFILES[NAME]
MANIFEST = os.path.join(ROOT, 'results/data/manifest.tsv')
SEEDS = os.path.join(HERE, 'seeds.yaml')
DESIGN = os.path.join(P['scripts'], 'design.yaml')
TASK_GENERATOR = os.path.join(P['scripts'], 'make_task_files.py')


def task_generator():
    """The project's make_task_files module (TASKS, budget_of, ...)."""
    spec = importlib.util.spec_from_file_location(f'make_task_files_{NAME}', TASK_GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
