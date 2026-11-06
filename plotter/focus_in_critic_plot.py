"""critic_no_critic.py's training-curve figure with the main-table FOCUS-OT-L2 / FOCUS-OT-Cos
runs added as extra curves (same 25-point smoothing), written to a new file."""

import critic_no_critic as cnc
from focus_vs_critic_ablation import load_main_runs

SMOOTH_WINDOW = 25
EXTRA = {"FOCUS-OT-L2 (main)": ("FOCUS-ot-l2", "#000000"), "FOCUS-OT-Cos (main)": ("FOCUS-ot-cos", "#6b6b6b")}


def main():
    data = cnc.build_data(critic_reward_shapping=True, x_axis="steps", smooth_window=SMOOTH_WINDOW)

    for label, (algo, color) in EXTRA.items():
        # cnc.METHOD_ORDER / ALGOS / COLOR are module-level containers used by the plotting code.
        cnc.METHOD_ORDER.append(label)
        cnc.ALGOS.insert(len(cnc.ALGOS) - 1, label)  # keep "Expert" last
        cnc.COLOR[label] = color
        for env, env_data in data.items():
            runs = []
            for run in load_main_runs(env, algo):
                ep_rew, steps, times = cnc.fetch_run_history(run["project"], run["run_id"])
                clipped = cnc.clip_to_env_cap(ep_rew, steps, times, env, x_axis="steps")
                if clipped is not None:
                    runs.append(clipped)
            built = cnc.interpolate_group(runs, n_points=1000, smooth_window=SMOOTH_WINDOW)
            if built is None:
                env_data[label] = {"data": [], "x": None}
            else:
                curves, grid = built
                env_data[label] = {"data": curves, "x": grid}

    out = cnc.FOLDER_TO_SAVE_PLOTS / "critic_no_critic_with_focus_main_mean95ci_steps_normalized.png"
    cnc.plot_training_curves(data, out, x_axis="steps", center_stat="mean", band="95ci", normalize_expert=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
