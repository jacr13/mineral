"""Verify per-pair aggregates and generate manuscript tables from rerun outputs."""
import csv
import json
from pathlib import Path
import struct
import hashlib
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from phase_mismatch_diagnostic import ENV_DEFAULTS
from paired_phase_rescue_diagnostic import cluster_bootstrap_mean_ci

# Match the float32 threshold used in the diagnostic tensor comparisons.
tolerance = struct.unpack("f", struct.pack("f", .1))[0]

root = Path('workdir/phase_rescue_verified_20260919')
dest = Path('docs/phase_rescue')
metrics = ['best_of_k_candidate_coverage_fraction', 'rescue_fraction_all',
           'phase_improved_fraction', 'k1_phase_error_mean', 'best_of_k_phase_error_mean']
labels = {'hopper': 'Hopper', 'ant': 'Ant', 'humanoid': 'Humanoid', 'snu_humanoid': 'SNU Humanoid'}
rows = []
summary = []
provenance = []
checksums = json.loads((dest / "sha256.json").read_text())
for name, expected in checksums.items():
    assert hashlib.file_digest(Path(name).open("rb"), "sha256").hexdigest() == expected, name
for env, label in labels.items():
    data = json.loads((root / env / 'paired_results.json').read_text())
    m = data['metadata']
    assert (m['environment'], m['ot_cost'], m['input_type'], m['query_source']) == (env, 'cosine', 'state_state', 'held_out_expert')
    assert (m['query_windows'], m['sampling_repeats'], m['query_trajectories'], m['phase_tolerance']) == (512, 4, 32, .1)
    expert = Path(ENV_DEFAULTS[env]['expert_path'])
    assert Path(m['expert_path']).resolve() == expert.resolve()
    assert m['criterion'] == 'ot' and m['reference_trajectories'] == 8
    assert m['arguments']['env'] == env and m['arguments']['ot_cost'] == 'cosine'
    r = next(r for r in data['results'] if r['target_mismatch'] == .5)
    with (root / env / 'paired_per_query.csv').open() as file:
        details = [d for d in csv.DictReader(file) if float(d['target_mismatch']) == .5]
    assert len(details) == 2048
    assert len({d['query_trajectory'] for d in details}) == 32
    for d in details:
        errors = json.loads(d['candidate_phase_errors'])
        costs = json.loads(d['candidate_costs'])
        slot = int(d['selected_slot'])
        assert slot == min(range(8), key=costs.__getitem__)
        assert (d['rescued'] == 'True') == (errors[0] > tolerance and errors[slot] <= tolerance)
        assert d['base_close'] == 'False'
    values = [sum(d['close_candidate_available'] == 'True' for d in details)/2048,
              sum(d['rescued'] == 'True' for d in details)/2048,
              sum(float(d['selected_phase_error']) < float(d['base_phase_error']) for d in details)/2048,
              sum(float(d['base_phase_error']) for d in details)/2048,
              sum(float(d['selected_phase_error']) for d in details)/2048]
    for key, value in zip(metrics, values):
        assert abs(r[key] - value) < 1e-6, (env, key)
        assert r[key + '_ci95_low'] <= r[key] <= r[key + '_ci95_high']
    # Recompute every displayed interval from the saved pair-level observations.
    cluster_ids = torch.tensor([int(d['query_trajectory']) for d in details])
    assert sorted(cluster_ids.unique().tolist()) == list(range(32))
    observations = [
        torch.tensor([d['close_candidate_available'] == 'True' for d in details]).float(),
        torch.tensor([d['rescued'] == 'True' for d in details]).float(),
        torch.tensor([float(d['selected_phase_error']) < float(d['base_phase_error']) for d in details]).float(),
        torch.tensor([float(d['base_phase_error']) for d in details]),
        torch.tensor([float(d['selected_phase_error']) for d in details]),
    ]
    for key, values in zip(metrics, observations):
        low, high = cluster_bootstrap_mean_ci(values, cluster_ids,
            num_samples=m['bootstrap_samples'], seed=m['bootstrap_seed'])
        assert abs(low - r[key + '_ci95_low']) < 1e-7, (env, key, 'low')
        assert abs(high - r[key + '_ci95_high']) < 1e-7, (env, key, 'high')
    artifacts = [root / env / name for name in ['paired_results.json', 'paired_per_query.csv', 'command.txt']]
    provenance.append({
        'environment': env, 'criterion': m['criterion'], 'ot_cost': m['ot_cost'],
        'expert_path': str(expert), 'expert_sha256': checksums[str(expert)],
        'reference_demonstration_indices': list(range(8)),
        'query_demonstration_indices': list(range(8,40)),
        'windows_per_query_trajectory': {str(i): int((cluster_ids == i).sum()) // 4 for i in range(32)},
        'baseline_error_values': sorted(set(observations[3].tolist())),
        'table_intervals_recomputed_from_per_query_csv': True,
        'artifact_sha256': {str(path): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() for path in artifacts},
    })
    def cell(key, percent=False):
        scale, digits = (100, 1) if percent else (1, 3)
        return f"{r[key]*scale:.{digits}f} [{r[key+'_ci95_low']*scale:.{digits}f}, {r[key+'_ci95_high']*scale:.{digits}f}]"
    rows.append((label, m['estimated_period'], [cell(k, i < 3) for i,k in enumerate(metrics)]))
    summary.append({'environment': env, 'metadata': m, 'half_cycle_results': r})
(dest / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
(dest / 'verified_results.json').write_text(json.dumps(summary, indent=2)+'\n')
tex = r'''\begin{table}[H]
\centering\small
\setlength{\tabcolsep}{0.35em}
\begin{tabular}{lccc}
\toprule
Environment & Coverage (\%) & Candidate rescued (\%) & Phase improved (\%) \\
\midrule
'''
for label, period, cells in rows:
    tex += label + ' & ' + ' & '.join(cells[:3]) + r' \\' + '\n'
tex += r'''\bottomrule
\end{tabular}
\caption{Paired phase-rescue diagnostic at a target half-cycle mismatch.
Entries are point estimates [95\% trajectory-cluster bootstrap CI].
All percentages use all 2048 query--pool pairs per environment.}
\label{tab:phase_rescue}
\end{table}
\begin{table}[H]
\centering\small
\begin{tabular}{lrcc}
\toprule
Environment & Period $P$ (steps) & Phase error $K=1$ & Phase error $K=8$ \\
\midrule
'''
for label, period, cells in rows:
    tex += f'{label} & {period} & ' + ' & '.join(cells[3:]) + r' \\' + '\n'
tex += r'''\bottomrule
\end{tabular}
\caption{Mean circular phase errors in gait cycles [95\% trajectory-cluster
bootstrap CI]. Phase labels use a single shared reference-cycle template
per environment.}
\label{tab:phase_rescue_errors}
\end{table}
'''
(dest / 'tables.tex').write_text(tex)
appendix = (dest / 'appendix.tex').read_text()
sweep = (dest / 'k_sweep_appendix.tex').read_text().replace('\\input{docs/phase_rescue/k_sweep_table.tex}', (dest / 'k_sweep_table.tex').read_text())
appendix = appendix.replace('\\input{docs/phase_rescue/k_sweep_appendix.tex}', sweep)
(dest / 'appendix_standalone.tex').write_text(appendix.replace('% Copy the generated tables.tex here when inserting this section into the manuscript.\n\\input{docs/phase_rescue/tables.tex}', tex))
for label, period, cells in rows:
    print(label, 'P=', period, *cells, sep=' | ')
print('Verified data/code hashes, environment/cosine provenance, all 8192 half-cycle pairs, and all 20 table confidence intervals recomputed from CSV.')
