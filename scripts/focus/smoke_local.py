"""Run task files locally with a tiny budget (smoke test): python smoke_local.py <task file> ... (from the checkout)."""
import os, subprocess, sys, tempfile, yaml
sys.path.insert(0, os.getcwd())
import spawner
SCR = os.environ.get('SMOKE_DIR', tempfile.gettempdir())
for path in sys.argv[1:]:
    cfg = yaml.safe_load(open(path))
    for k in ('sweep', 'spawner'):
        cfg.pop(k, None)
    cfg['agent']['shac']['max_agent_steps'] = int(os.environ.get('SMOKE_STEPS', 20480))
    cfg['agent']['eval_every_steps'] = int(os.environ.get('SMOKE_EVAL', 8192))
    cfg['logdir'] = tempfile.mkdtemp(prefix='smoke_', dir=SCR)
    ov = [o.strip() for o in spawner._build_overrides(cfg, 'local', indent='')]
    r = subprocess.run(['uv', 'run', '--locked', 'python', '-m', 'mineral.scripts.run'] + ov, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print('==', path, 'rc', r.returncode, cfg['logdir'])
    print('\n'.join([l for l in out.splitlines() if any(k in l for k in ('periodic eval', 'Exception', 'Error', 'Traceback', 'eval_scores', 'Agent Steps'))][-12:]))
    print(sorted(os.listdir(cfg['logdir'])))
