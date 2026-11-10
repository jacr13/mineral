"""Check ablation commands and queue limits without submitting cluster jobs."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/focus_gradpath.sh'


class FocusGradpathTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.environment = {
            **os.environ,
            'ENVS': 'dflex_hopper',
            'ARMS': 'true false',
            'DRY_RUN': '1',
            'MAX_JOBS': '100',
            'BATCH_JOBS': '6',
            'RUNTIME': '1h',
        }

    def run_script(self, **overrides):
        return subprocess.run(
            ['bash', str(SCRIPT)],
            cwd=self.directory,
            env={**self.environment, **overrides},
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_dry_run_uses_uv_from_outside_repository(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('uv ')]
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[:5], ['uv', 'run', '--locked', 'python', 'spawner.py'])
            self.assertEqual(command[command.index('--sweep_max') + 1], '6')
        self.assertIn('no jobs submitted', result.stdout)

    def test_invalid_inputs_fail_before_submission(self):
        for overrides in (
            {'ENVS': 'dflex_hopper typo'},
            {'ENVS': ' '},
            {'ARMS': 'true typo'},
            {'ARMS': ' '},
            {'DRY_RUN': 'true'},
            {'MAX_JOBS': '5'},
            {'MAX_JOBS': 'invalid'},
            {'BATCH_JOBS': '3'},
        ):
            with self.subTest(overrides=overrides):
                result = self.run_script(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('ERROR:', result.stderr)
                self.assertNotIn('arm:', result.stdout)

    def test_queue_reserves_full_batch_when_other_jobs_finish(self):
        self.install_fake_scheduler([0, 3, 3, 0, 6])
        result = self.run_script(DRY_RUN='0', MAX_JOBS='6')
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        submissions = [event for event in events if event['tool'] == 'uv']
        self.assertEqual(len(submissions), 2)
        self.assertTrue(all(event['cwd'] == str(ROOT) for event in submissions))
        queues = [event for event in events if event['tool'] == 'squeue']
        self.assertEqual(len(queues), 5)
        self.assertTrue(all('-r' in event['args'] for event in queues))
        self.assertIn({'tool': 'sleep', 'args': ['60'], 'cwd': str(ROOT)}, events)
        self.assertIn('Queue 3 > 0 (need room for 6)', result.stdout)

    def test_queue_failure_does_not_submit(self):
        self.install_fake_scheduler([])
        result = self.run_script(DRY_RUN='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(event['tool'] == 'uv' for event in self.events()))

    def install_fake_scheduler(self, queue_counts):
        bin_dir = self.directory / 'bin'
        bin_dir.mkdir()
        self.environment.update(
            PATH=f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
            TEST_STATE=str(self.directory),
        )
        (self.directory / 'queue.json').write_text(json.dumps(queue_counts))
        program = (
            f'#!{sys.executable}\n'
            + '''import json
import os
import sys
from pathlib import Path

state = Path(os.environ['TEST_STATE'])
tool = Path(sys.argv[0]).name
with (state / 'events.jsonl').open('a') as log:
    log.write(json.dumps({'tool': tool, 'args': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')
if tool == 'squeue':
    queue_file = state / 'queue.json'
    counts = json.loads(queue_file.read_text())
    if not counts:
        sys.exit(1)
    count = counts.pop(0)
    queue_file.write_text(json.dumps(counts))
    for job in range(count):
        print(job)
'''
        )
        for name in ('uv', 'squeue', 'sleep'):
            executable = bin_dir / name
            executable.write_text(program)
            executable.chmod(0o755)

    def events(self):
        return [json.loads(line) for line in (self.directory / 'events.jsonl').read_text().splitlines()]

    def test_generated_arms_have_matching_seeds_and_requested_runtime(self):
        (self.directory / 'tasks').symlink_to(ROOT / 'tasks', target_is_directory=True)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('uv ')]
        self.assertEqual(len(commands), 2)
        for command in commands:
            arguments = [arg for arg in command[5:] if arg != '--deploy_now']
            generated = subprocess.run(
                [sys.executable, str(ROOT / 'spawner.py'), *arguments, '--no-deploy_now'],
                cwd=self.directory,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
        scripts = list((self.directory / 'spawn').rglob('*.sh'))
        self.assertEqual(len(scripts), 12)
        self.assertTrue((self.directory / 'out').is_dir())
        seeds = {'true': set(), 'false': set()}
        for script in scripts:
            content = script.read_text()
            self.assertIn('#SBATCH --time=0-01:00:00', content)
            self.assertIn('max_runtime=1h', content)
            self.assertIn('agent=FOCUS/DFlexAntSAPO', content)
            arm = 'false' if 'agent.focus.loss_ot_detach_plan=false' in content else 'true'
            seed = next(
                line.strip().split('=')[1].split()[0] for line in content.splitlines() if line.strip().startswith('seed=')
            )
            seeds[arm].add(int(seed))
        self.assertEqual(seeds['true'], {100, 110, 120, 1000, 1100, 1200})
        self.assertEqual(seeds['false'], seeds['true'])

    def test_runtime_defaults_to_task_setting_then_twelve_hours(self):
        task = self.directory / 'task.yaml'
        for task_setting, expected in (('spawner:\n  runtime: 8h\n', '08'), ('', '12')):
            with self.subTest(task_setting=task_setting):
                task.write_text('task:\n  name: DFlex\n' + task_setting)
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / 'spawner.py'),
                        '--task_name',
                        str(task),
                        '--env_files',
                        task.name,
                        '--deployment',
                        'slurm',
                        '--no-deploy_now',
                    ],
                    cwd=self.directory,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                scripts = list((self.directory / 'spawn').rglob('*.sh'))
                self.assertEqual(len(scripts), 1)
                self.assertIn(f'#SBATCH --time=0-{expected}:00:00', scripts[0].read_text())
                scripts[0].unlink()


if __name__ == '__main__':
    unittest.main()
