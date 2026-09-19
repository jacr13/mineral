#!/usr/bin/env python3
"""Extend verified half-cycle K=8 pools to nested K=4/8/16 pools.

Run from the repository root with OMP_NUM_THREADS=2 MKL_NUM_THREADS=2.
The first four candidates of each verified K=8 pool define K=4; K=16
appends eight independent uniform draws (seed 67890). Original K=8 scores
are recomputed and checked before reuse. No training is performed.
"""
import csv
import hashlib
import json
from pathlib import Path

import torch
import paired_phase_rescue_diagnostic as paired
import phase_mismatch_diagnostic as diagnostic

SOURCE = Path('workdir/phase_rescue_verified_20260919')
OUTPUT = Path('workdir/phase_rescue_k_sweep_20260919')
REPORT = Path('docs/phase_rescue')
ENVIRONMENTS = ['hopper', 'ant', 'humanoid', 'snu_humanoid']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(env):
    source = SOURCE / env
    previous = json.loads((source / 'paired_results.json').read_text())
    metadata = previous['metadata']
    assert (metadata['environment'], metadata['ot_cost'], metadata['input_type']) == (env, 'cosine', 'state_state')
    hashes = json.loads((REPORT / 'sha256.json').read_text())
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, path
    args = paired.build_parser().parse_args(['--env', env])
    for key, value in metadata['arguments'].items():
        setattr(args, key, Path(value) if key in ['expert_path', 'rollout_path', 'output_dir'] and value else value)
    args.device = 'cpu'
    args.generator = torch.Generator(device='cpu').manual_seed(args.seed)
    torch.manual_seed(args.seed)
    data = paired.prepare_data(args)
    with (source / 'paired_per_query.csv').open() as stream:
        old = [row for row in csv.DictReader(stream) if float(row['target_mismatch']) == .5]
    assert len(old) == 2048
    lookup = {(int(t), int(s)): i for i, (t, s) in enumerate(zip(data['reference_trajectory_ids'], data['reference_starts']))}
    candidates = []
    for row in old:
        q = int(row['query_id'])
        assert int(data['query_trajectory_ids'][q]) == int(row['query_trajectory'])
        assert int(data['query_starts'][q]) == int(row['query_start'])
        assert int(data['oracle_phases'][q]) == int(row['oracle_phase'])
        candidates.append([lookup[(t, s)] for t, s in zip(json.loads(row['candidate_reference_trajectories']), json.loads(row['candidate_reference_starts']))])
    original = torch.tensor(candidates)
    qids = torch.tensor([int(row['query_id']) for row in old])
    repeat_ids = torch.tensor([int(row['sampling_repeat']) for row in old])
    queries = data['query_features'][qids]
    criterion = diagnostic.build_criterion(args)
    old_costs = torch.tensor([json.loads(row['candidate_costs']) for row in old])
    recomputed = diagnostic.candidate_costs(queries, data['reference_features'], original, criterion, args.criterion_batch_size)
    torch.testing.assert_close(recomputed, old_costs, rtol=1e-5, atol=1e-6)
    extra = torch.randint(len(data['reference_features']), (len(old), 8), generator=torch.Generator().manual_seed(67890))
    all_candidates = torch.cat([original, extra], dim=1)
    extra_costs = diagnostic.candidate_costs(queries, data['reference_features'], extra, criterion, args.criterion_batch_size)
    all_costs = torch.cat([old_costs, extra_costs], dim=1)
    oracle_indices = data['oracle_indices'][qids]
    oracle_costs = diagnostic.candidate_costs(queries, data['reference_features'], oracle_indices[:, None], criterion, args.criterion_batch_size)[:, 0]
    summaries, details, by_k = [], [], {}
    for k in [4, 8, 16]:
        args.k = k
        summary, records = paired.paired_summary(.5, all_candidates[:, :k], all_costs[:, :k], data, args, qids, repeat_ids, oracle_indices, data['oracle_costs'][qids], data['oracle_phases'][qids], oracle_costs)
        if k == 8:
            previous_row = next(r for r in previous['results'] if r['target_mismatch'] == .5)
            for key, value in previous_row.items():
                if isinstance(value, (float, int)):
                    assert abs(summary[key] - value) < 1e-7, (key, summary[key], value)
        for record in records:
            record['k'] = k
        summaries.append(summary)
        details.extend(records)
        by_k[k] = records
    # Within-pair differences retain the same trajectory clusters across K.
    cluster_ids = data['query_trajectory_ids'][qids]
    differences = []
    for smaller, larger in [(4, 8), (8, 16), (4, 16)]:
        for metric, field in [('coverage', 'close_candidate_available'), ('rescue', 'rescued'), ('phase_error', 'selected_phase_error')]:
            values = torch.tensor([float(b[field]) - float(a[field]) for a, b in zip(by_k[smaller], by_k[larger])])
            low, high = paired.cluster_bootstrap_mean_ci(values, cluster_ids, num_samples=args.bootstrap_samples, seed=args.bootstrap_seed)
            differences.append({'smaller_k': smaller, 'larger_k': larger, 'metric': metric, 'difference_larger_minus_smaller': float(values.mean()), 'ci95_low': low, 'ci95_high': high})
    out = OUTPUT / env
    out.mkdir(parents=True, exist_ok=True)
    paired.write_csv(out / 'paired_per_query.csv', details)
    paired.write_csv(out / 'paired_summary.csv', summaries)
    result = {'environment': env, 'source_metadata': metadata, 'device': 'cpu', 'k_values': [4, 8, 16], 'extra_candidate_seed': 67890, 'nesting': 'K4 = first four original K8 slots; K16 = original K8 plus eight uniform reference-window draws', 'source_sha256': {str(source / name): digest(source / name) for name in ['paired_results.json', 'paired_per_query.csv']}, 'script_sha256': digest(Path(__file__)), 'original_k8_costs_recomputed_and_verified': True, 'original_k8_summary_and_intervals_reproduced': True, 'results': summaries, 'paired_differences': differences}
    (out / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def report(results):
    (REPORT / 'k_sweep_results.json').write_text(json.dumps(results, indent=2)+'\n')
    text = ['# Nested K comparison', '', 'Point estimates [95% trajectory-cluster bootstrap CI]; coverage and rescue are percentages, phase error is in gait cycles.', '', '| Environment | K | Coverage | Rescue | Mean phase error |', '|---|---:|---:|---:|---:|']
    tex = [r'\begin{table}[H]', r'\centering\small', r'\begin{tabular}{lrccc}', r'\toprule', r'Environment & $K$ & Coverage (\%) & Rescue (\%) & Mean phase error \\', r'\midrule']
    for result in results:
        for row in result['results']:
            cells = []
            for key, scale, digits in [('best_of_k_candidate_coverage_fraction', 100, 1), ('rescue_fraction_all', 100, 1), ('best_of_k_phase_error_mean', 1, 3)]:
                cells.append(f"{row[key]*scale:.{digits}f} [{row[key+'_ci95_low']*scale:.{digits}f}, {row[key+'_ci95_high']*scale:.{digits}f}]")
            text.append('| ' + ' | '.join([result['environment'], str(row['k']), *cells]) + ' |')
            tex.append(' & '.join([result['environment'].replace('_', r'\_'), str(row['k']), *cells]) + r' \\')
    text.extend(['', 'Paired changes are larger K minus smaller K; coverage/rescue differences are percentage points. Phase-error decreases are improvements. Intervals are pointwise, with no multiplicity adjustment.', '', '| Environment | Comparison | Metric | Paired change [95% CI] |', '|---|---|---|---:|'])
    for result in results:
        for row in result['paired_differences']:
            scale, digits = (1, 3) if row['metric'] == 'phase_error' else (100, 1)
            cell = f"{row['difference_larger_minus_smaller']*scale:.{digits}f} [{row['ci95_low']*scale:.{digits}f}, {row['ci95_high']*scale:.{digits}f}]"
            text.append(f"| {result['environment']} | {row['smaller_k']} to {row['larger_k']} | {row['metric']} | {cell} |")
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\caption{Nested-pool phase-rescue comparison at a target half-cycle mismatch. Entries are point estimates [95\% trajectory-cluster bootstrap CI], using 512 held-out query windows and four pools per window. $K=4$ uses the first four candidates of the verified $K=8$ pool; $K=16$ appends eight uniformly sampled candidates. All conditions share the same mismatched baseline.}', r'\label{tab:phase_rescue_k_sweep}', r'\end{table}'])
    (REPORT / 'k_sweep.md').write_text('\n'.join(text)+'\n')
    (REPORT / 'k_sweep_table.tex').write_text('\n'.join(tex)+'\n')


if __name__ == '__main__':
    results = []
    for env in ENVIRONMENTS:
        print('Running', env, flush=True)
        results.append(run(env))
        print('Completed', env, flush=True)
    report(results)
