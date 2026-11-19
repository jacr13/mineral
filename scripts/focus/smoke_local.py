"""Run task files locally with a tiny budget through the rerun protocol (smoke test).

usage (from the checkout): python scripts/focus/smoke_local.py <task file> ...
env: SMOKE_STEPS (budget, default 20480), SMOKE_EVAL (eval_every_steps, default 8192), SMOKE_DIR (output root).
"""
import json, os, subprocess, sys, tempfile, yaml
sys.path.insert(0, os.getcwd())
import spawner
SCR = os.environ.get('SMOKE_DIR', tempfile.gettempdir())
STEPS = int(os.environ.get('SMOKE_STEPS', 20480))
for path in sys.argv[1:]:
    cfg = yaml.safe_load(open(path))
    for k in ('sweep', 'spawner'):
        cfg.pop(k, None)
    ag = cfg.setdefault('agent', {})
    for fam in ('shac', 'ppo', 'sac', 'ddpg'):
        if isinstance(ag.get(fam), dict) and 'max_agent_steps' in ag[fam]:
            ag[fam]['max_agent_steps'] = STEPS
    if isinstance(ag.get('diffmimic'), dict):
        ag['diffmimic']['max_agent_steps'] = STEPS
    for fam in ('bc',):
        if isinstance(ag.get(fam), dict):
            ag[fam]['rounds'] = min(int(ag[fam].get('rounds', 2)), 2)
    ag['protocol_eval'] = True
    ag['eval_every_steps'] = int(os.environ.get('SMOKE_EVAL', 8192))
    ag['ckpt_every'] = -1
    cfg['logdir'] = tempfile.mkdtemp(prefix='smoke_', dir=SCR)
    cfg['wandb'] = {'mode': 'disabled'}
    cfg['run'] = 'train_eval'
    ov = [o.strip() for o in spawner._build_overrides(cfg, 'local', indent='')]
    r = subprocess.run(['uv', 'run', '--locked', 'python', '-m', 'mineral.scripts.run'] + ov, capture_output=True, text=True)
    out = r.stdout + r.stderr
    d = cfg['logdir']
    res = {}
    for f in ('scores.json', 'scores_stochastic.json'):
        if os.path.exists(os.path.join(d, f)):
            s = json.load(open(os.path.join(d, f)))
            res[f] = (s.get('eval_scores/sample'), s.get('agent_steps'), len(s['eval_scores/episode_rewards']),
                      round(sum(s['eval_scores/episode_rewards']) / max(1, len(s['eval_scores/episode_rewards'])), 1))
    curve = [json.loads(l)['agent_steps'] for l in open(os.path.join(d, 'eval_curve.jsonl'))] if os.path.exists(os.path.join(d, 'eval_curve.jsonl')) else None
    budget = json.load(open(os.path.join(d, 'budget.json'))) if os.path.exists(os.path.join(d, 'budget.json')) else None
    err = [l for l in out.splitlines() if 'Traceback' in l or 'Error' in l or 'Exception' in l][-3:]
    print('==', path, 'rc', r.returncode, d)
    print('   scores', res, '\n   curve', curve, '\n   budget', budget and {k: budget[k] for k in ('agent_steps', 'max_agent_steps', 'reached_budget')}, '\n   err', err)
