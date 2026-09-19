# Nested K comparison

Point estimates [95% trajectory-cluster bootstrap CI]; coverage and rescue are percentages, phase error is in gait cycles.

| Environment | K | Coverage | Rescue | Mean phase error |
|---|---:|---:|---:|---:|
| hopper | 4 | 47.3 [44.9, 49.7] | 32.6 [30.2, 35.0] | 0.200 [0.190, 0.211] |
| hopper | 8 | 78.2 [76.5, 79.9] | 46.3 [43.6, 48.9] | 0.147 [0.139, 0.156] |
| hopper | 16 | 95.8 [94.9, 96.6] | 56.6 [54.1, 58.8] | 0.118 [0.112, 0.124] |
| ant | 4 | 51.8 [49.8, 53.9] | 40.1 [38.0, 42.3] | 0.168 [0.162, 0.174] |
| ant | 8 | 82.0 [80.2, 83.8] | 63.4 [60.9, 65.9] | 0.102 [0.098, 0.107] |
| ant | 16 | 97.7 [97.2, 98.3] | 81.4 [79.6, 83.3] | 0.073 [0.069, 0.078] |
| humanoid | 4 | 56.1 [54.4, 57.9] | 43.0 [41.3, 44.8] | 0.190 [0.184, 0.196] |
| humanoid | 8 | 85.2 [83.8, 86.7] | 64.0 [62.1, 65.9] | 0.121 [0.116, 0.125] |
| humanoid | 16 | 98.0 [97.4, 98.7] | 77.7 [76.0, 79.3] | 0.089 [0.085, 0.093] |
| snu_humanoid | 4 | 55.8 [53.7, 58.0] | 48.0 [45.9, 50.4] | 0.168 [0.163, 0.174] |
| snu_humanoid | 8 | 85.1 [83.8, 86.5] | 72.0 [70.1, 73.9] | 0.095 [0.092, 0.099] |
| snu_humanoid | 16 | 97.9 [97.4, 98.5] | 87.2 [85.6, 88.8] | 0.058 [0.055, 0.062] |

Paired changes are larger K minus smaller K; coverage/rescue differences are percentage points. Phase-error decreases are improvements. Intervals are pointwise, with no multiplicity adjustment.

| Environment | Comparison | Metric | Paired change [95% CI] |
|---|---|---|---:|
| hopper | 4 to 8 | coverage | 30.9 [28.6, 33.2] |
| hopper | 4 to 8 | rescue | 13.8 [11.1, 16.4] |
| hopper | 4 to 8 | phase_error | -0.053 [-0.063, -0.044] |
| hopper | 8 to 16 | coverage | 17.5 [15.9, 19.2] |
| hopper | 8 to 16 | rescue | 10.3 [8.5, 12.3] |
| hopper | 8 to 16 | phase_error | -0.029 [-0.034, -0.025] |
| hopper | 4 to 16 | coverage | 48.4 [46.3, 50.6] |
| hopper | 4 to 16 | rescue | 24.0 [21.0, 26.9] |
| hopper | 4 to 16 | phase_error | -0.082 [-0.093, -0.072] |
| ant | 4 to 8 | coverage | 30.2 [28.2, 32.1] |
| ant | 4 to 8 | rescue | 23.3 [20.4, 26.3] |
| ant | 4 to 8 | phase_error | -0.066 [-0.071, -0.060] |
| ant | 8 to 16 | coverage | 15.7 [14.2, 17.4] |
| ant | 8 to 16 | rescue | 18.0 [16.4, 19.7] |
| ant | 8 to 16 | phase_error | -0.029 [-0.032, -0.026] |
| ant | 4 to 16 | coverage | 45.9 [43.9, 47.9] |
| ant | 4 to 16 | rescue | 41.3 [38.6, 44.0] |
| ant | 4 to 16 | phase_error | -0.095 [-0.101, -0.089] |
| humanoid | 4 to 8 | coverage | 29.1 [27.4, 30.8] |
| humanoid | 4 to 8 | rescue | 21.0 [19.0, 23.1] |
| humanoid | 4 to 8 | phase_error | -0.069 [-0.074, -0.064] |
| humanoid | 8 to 16 | coverage | 12.8 [11.4, 14.2] |
| humanoid | 8 to 16 | rescue | 13.7 [12.0, 15.4] |
| humanoid | 8 to 16 | phase_error | -0.032 [-0.036, -0.028] |
| humanoid | 4 to 16 | coverage | 41.9 [40.2, 43.6] |
| humanoid | 4 to 16 | rescue | 34.7 [32.2, 37.1] |
| humanoid | 4 to 16 | phase_error | -0.101 [-0.108, -0.094] |
| snu_humanoid | 4 to 8 | coverage | 29.3 [27.3, 31.2] |
| snu_humanoid | 4 to 8 | rescue | 24.0 [21.5, 26.3] |
| snu_humanoid | 4 to 8 | phase_error | -0.073 [-0.078, -0.068] |
| snu_humanoid | 8 to 16 | coverage | 12.8 [11.5, 14.1] |
| snu_humanoid | 8 to 16 | rescue | 15.1 [13.2, 17.1] |
| snu_humanoid | 8 to 16 | phase_error | -0.037 [-0.041, -0.033] |
| snu_humanoid | 4 to 16 | coverage | 42.1 [39.9, 44.3] |
| snu_humanoid | 4 to 16 | rescue | 39.1 [36.6, 41.4] |
| snu_humanoid | 4 to 16 | phase_error | -0.110 [-0.116, -0.104] |
