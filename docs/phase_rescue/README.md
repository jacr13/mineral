# Verified phase-rescue diagnostic

The fresh CPU rerun reproduces every manuscript point estimate at the reported precision. Previous directories were unreliable: `hopper_ot_cos` contained Ant metadata, and `snu_humanoid_ot_cos` contained an L2 run. The new outputs identify the correct environment and cosine cost for all four environments.

Use `appendix_standalone.tex` as a complete replacement subsection. It includes both tables, the gait-period estimator, the shared template construction, metric definitions, and 95% trajectory-cluster bootstrap intervals. `appendix.tex` and `tables.tex` are the editable source components. No manuscript build was available here.

Run from the repository root:

```bash
python docs/phase_rescue/rerun.py
python docs/phase_rescue/report.py
```

Raw outputs, exact commands, logs, plots, and per-pair CSV files are under `workdir/phase_rescue_verified_20260919/<environment>/`. `verified_results.json` preserves the half-cycle summaries and full run metadata in this documentation directory. The rerun retains the original five-offset sequence because preceding draws affect the random pools at the final half-cycle offset. Restricting the run to only 0.5 would draw different pools.

Configuration: first eight demonstrations as references, next 32 as held-out expert queries; 512 windows, four repeats, 32 transitions, K=8, unstandardized state-transition features, OT cosine cost, epsilon 0.1, 60 Sinkhorn iterations, phase tolerance 0.10 cycles, mismatch bin half-width 0.04 cycles, sampling seed 0. Bootstrap: 5000 resamples, seed 12345, complete query trajectories retained as clusters. These are conditional intervals with the reference bank and template fixed; they are not training-seed or learned-policy performance intervals.

Period estimates are 37, 23, 20, and 30 steps for Hopper, Ant, Humanoid, and SNU Humanoid. Each environment uses a single cycle from reference demonstration zero starting at timestep 100. No per-demonstration templates are constructed. The selected minimum-cost candidate is the diagnostic outcome; the results do not claim equivalent improvement for the soft mixture.

Validation: the report checks environment and criterion metadata, all 8192 half-cycle pairs, 32 query-trajectory clusters per environment, minimum-cost selection, rescue flags, and table aggregates against the raw CSV files. Targeted checks also verified circular wraparound, constant/single-cluster bootstrap cases, and propagation of OT epsilon/iterations. All four original rows reproduce at their displayed precision. CPU PyTorch was used because the installed CUDA build does not support the local GTX 1070; the failed GPU attempt produced no results used in these tables.

Intervals with equal displayed endpoints reflect constant errors or rounding to three decimals. Percentages use all pairs, including pools without compatible candidates. The original code's float32 comparison is retained at the compatibility boundary.

## Provenance audit and manuscript wording

`provenance.json` records the environment, OT cosine setting, expected expert-file hash, raw-result/CSV/command hashes, demonstration indices, sampled-window counts for every cluster, and observed baseline error values. `report.py` verifies input and code hashes and recomputes all 20 displayed confidence intervals directly from the corresponding environment's per-query CSV. Both tables are generated from those checked summaries; none uses the old mislabeled directories.

All 32 held-out demonstrations (original indices 8–39; local query IDs 0–31) contributed sampled windows in every environment. The bootstrap samples 32 IDs with replacement per draw. The implementation excludes trajectories with no sampled windows; there were none in these runs. The 32 additional demonstrations serve this offline diagnostic only, beyond the stated eight-demonstration training setup.

`main_text.tex` contains a replacement claim limited to minimum-cost selection on held-out expert windows. The full manuscript is absent from this repository, so this is an insertion-ready replacement rather than a modification of the original main text. The appendix now defines coverage explicitly, specifies eight full transition vectors, and explains the fixed baseline errors for Hopper, Ant, and Humanoid.

## K=4, 8, and 16 extension

The half-cycle experiment now includes nested K=4/8/16 pools. K=4 takes the first four original K=8 candidates; K=16 appends eight uniform candidates with replacement using seed 67890. This preserves the exact original K=8 results and shared mismatched candidate. All original K=8 costs were recomputed and checked, and all original K=8 summary values and confidence intervals reproduce.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/phase_rescue_k_sweep.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python docs/phase_rescue/report.py
```

`k_sweep.md` contains coverage, rescue, phase error, and paired differences with pointwise 95% intervals. `k_sweep_results.json` includes all diagnostic metrics, including phase improvement, source metadata, source-file hashes, and paired bootstrap differences. Raw per-pair records are in `workdir/phase_rescue_k_sweep_20260919/`. `k_sweep_appendix.tex` and `k_sweep_table.tex` are incorporated into the complete `appendix_standalone.tex`.

Validation checked all 24,576 half-cycle records for nested candidate identities/costs, the common baseline, and minimum-cost selection; all 36 paired-difference intervals were recomputed from the saved CSVs. Increasing K improved average rescue and phase error in all four environments. This is evidence about minimum-cost selection on held-out expert windows, not a comparison of training outcomes. Only the half-cycle condition was extended to K=4 and K=16.
