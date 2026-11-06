"""critic_no_critic.py's curves and tables, with the two rows that are the FOCUS configurations
(critic-exp-l2 = FOCUS-OT-L2, critic-logexp-cos = FOCUS-OT-Cos) taken from the runs behind the
main table (enabled entries of grouped_exp_logdirs.yaml) instead of the reward-ablation sweep.
All other rows are unchanged. Same 25-point smoothing and per-seed-endpoint rule.
Output goes to a new folder; the original ablation outputs are not touched."""

import critic_no_critic as cnc
from focus_vs_critic_ablation import load_main_runs

SMOOTH_WINDOW = 25
REPLACE = {"critic-exp-l2": "FOCUS-ot-l2", "critic-logexp-cos": "FOCUS-ot-cos"}
cnc.FOLDER_TO_SAVE_PLOTS = cnc.PLOTTER_DIR / "crit_no_critic_focus_main_smoothed25"

_orig_build_data = cnc.build_data
_cache = {}


def build_data_with_focus_main(**kwargs):
    key = (kwargs.get("critic_reward_shapping"), kwargs.get("x_axis"))
    if key not in _cache:
        data = _orig_build_data(**kwargs)
        for env, env_data in data.items():
            for method, algo in REPLACE.items():
                runs = []
                for run in load_main_runs(env, algo):
                    ep_rew, steps, times = cnc.fetch_run_history(run["project"], run["run_id"])
                    clipped = cnc.clip_to_env_cap(ep_rew, steps, times, env, x_axis=kwargs.get("x_axis", "steps"))
                    if clipped is not None:
                        runs.append(clipped)
                built = cnc.interpolate_group(runs, n_points=kwargs.get("n_points", 1000), smooth_window=SMOOTH_WINDOW)
                curves, grid = built
                env_data[method] = {"data": curves, "x": grid}
                print(f"  {env}/{method}: replaced with {len(curves)} main-table FOCUS run(s)")
        _cache[key] = data
    return _cache[key]


cnc.build_data = build_data_with_focus_main

if __name__ == "__main__":
    for center_stat, band in [("mean", "std"), ("mean", "95ci"), ("median", "95ci")]:
        for normalize_expert in (True, False):
            cnc.main(
                critic_reward_shapping=True,
                x_axis="steps",
                center_stat=center_stat,
                band=band,
                smooth_window=SMOOTH_WINDOW,
                force_refresh=False,
                normalize_expert=normalize_expert,
            )
    print("Wrote", cnc.FOLDER_TO_SAVE_PLOTS)
