"""Run the four paired diagnostics from the repository root."""
import os
from pathlib import Path
import subprocess
import sys

root = Path('workdir/phase_rescue_verified_20260919')
for env in ['hopper', 'ant', 'humanoid', 'snu_humanoid']:
    out = root / env
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, 'scripts/paired_phase_rescue_diagnostic.py',
           '--env', env, '--input-type', 'state_state', '--criterion', 'ot',
           '--ot-cost', 'cosine', '--normalization', 'none',
           '--reference-trajectories', '8', '--query-trajectories', '32',
           '--num-query-windows', '512', '--sampling-repeats', '4',
           '--horizon', '32', '--mismatch-offsets', '0,0.125,0.25,0.375,0.5',
           '--phase-tolerance', '0.1', '--bootstrap-samples', '5000',
           '--bootstrap-seed', '12345', '--seed', '0', '--device', 'cpu',
           '--output-dir', str(out)]
    (out / 'command.txt').write_text(' '.join(cmd) + '\n')
    print('Running', env, flush=True)
    with (out / 'run.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    print('Completed', env, flush=True)
