from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

from utils import load_yaml, save_yaml, load_json, sync_exp_from_remote
from wandb_api import get_group_runs
import seaborn as sns

sns.set_theme()
sns.set(rc={"axes.facecolor": "#f5f5f5"})


EXPERTS = {
    'hopper': {'mean': 4812.96875, 'std': 7.324751853942871}, 'ant': {'mean': 9329.505859375, 'std': 37.58782196044922}, 'humanoid': {'mean': 8225.2177734375, 'std': 81.0524673461914}, 'snu_humanoid': {'mean': 6748.921875, 'std': 49.13310623168945},}

ENV_NAMES = {
    "hopper": "Hopper",
    "ant": "Ant",
    "humanoid": "Humanoid",
    "snu_humanoid": "SNU Humanoid"
}

COLOR = {
    "Expert": "#616161",
    "SAMfO/DACfO": "#A46750",
    "OPOLO": "#A12864",
    "GAIfO": "#7D54B2",
    "MAAD": "#DC4BDC",
    "ILD": "#FFB83D",
    "FOCUS-l2": "#5BC5DB",
    "FOCUS-OT-cos": "#A0C75C",
    "FOCUS-OT-l2": "#479A5F",
}

ALGOS = [
    "Expert",
    "SAMfO/DACfO",
    "OPOLO",
    "GAIfO",
    "MAAD",
    "ILD",
    "FOCUS-l2",
    "FOCUS-OT-l2",
    "FOCUS-OT-cos",
]

ALGO_DISPLAY = {
    "SAMfO/DACfO": "SAMfO/DACfO$^\\dag$",
    "OPOLO": "OPOLO$^\\dag$",
    "GAIfO": "GAIfO$^\\dag$",
}

ALGO_INDEX = {algo: idx for idx, algo in enumerate(ALGOS)}

import math
import numpy as np
import matplotlib.pyplot as plt

def normalize_algo_key(algo_key):
    key_lower = algo_key.lower()
    if key_lower.startswith("focus-ot-"):
        parts = algo_key.split("-")
        if len(parts) >= 3:
            return f"{parts[0]}-OT-{parts[2]}"
    return algo_key

def get_algo_display(algo_key, ALGO_DISPLAY):
    normalized = normalize_algo_key(algo_key)
    return ALGO_DISPLAY.get(normalized, normalized)

def algo_sort_key(algo_key, ALGO_INDEX, ALGO_DISPLAY):
    disp = get_algo_display(algo_key, ALGO_DISPLAY)
    return ALGO_INDEX.get(disp, ALGO_INDEX.get(normalize_algo_key(algo_key), 10**9))

def get_linestyle(algo_disp):
    if algo_disp == "Expert":
        return "--"
    # If you want special patterns like in your first script, add them here.
    # Example:
    # if algo_disp.startswith("OPOLO"): return "-."
    return "-"

def plot_results_like_first(
    data,
    output_path,
    COLOR,
    ALGOS,
    ALGO_DISPLAY,
    ALGO_INDEX,
    EXPERTS=None,
    *,
    normalize_expert=True,      # divide by expert mean per env
    add_expert_line=True,       # plot dashed expert line from EXPERTS
    shared_legend=True,         # one legend for the whole figure
    n_cols=4,
    ylim_bottom=0.0,
):
    """
    data[env][algo] = {
        "data": [np.array(n_points), ...]  # per-seed
        "agent_steps": [...],
        "epochs": [...]
    }
    """
    if not data:
        return

    env_names = list(data.keys())
    n_envs = len(env_names)
    n_cols = min(n_cols, n_envs) if n_envs > 1 else 1
    n_rows = math.ceil(n_envs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    # We will collect legend handles/labels from the first axis that has data,
    # then render a single legend at top (like your first script).
    legend_handles = None
    legend_labels = None

    for ax_idx, env_name in enumerate(env_names):
        ax = axes[ax_idx]
        env_data = data[env_name]

        # Determine common x
        n_points = None
        max_steps = 0
        for algo_data in env_data.values():
            if algo_data["data"] and n_points is None:
                n_points = algo_data["data"][0].shape[0]
            if algo_data.get("agent_steps"):
                max_steps = max(max_steps, max(algo_data["agent_steps"]))

        if n_points is None:
            ax.set_visible(False)
            continue

        if max_steps > 0:
            x = np.linspace(0, max_steps, num=n_points)
            x_label = "Steps"
        else:
            x = np.linspace(0.0, 1.0, num=n_points)
            x_label = "Training progress"

        # Expert mean for normalization / line
        expert_mean = None
        expert_std = None
        if EXPERTS and env_name in EXPERTS:
            expert_mean = float(EXPERTS[env_name]["mean"])
            expert_std = float(EXPERTS[env_name]["std"])

        # Sort algos in your desired global order
        algo_keys = sorted(
            env_data.keys(),
            key=lambda k: algo_sort_key(k, ALGO_INDEX, ALGO_DISPLAY),
        )

        # Optionally add explicit Expert dashed line even if not in env_data
        if add_expert_line and expert_mean is not None:
            y = np.full_like(x, expert_mean, dtype=np.float64)
            if normalize_expert and expert_mean != 0:
                y = y / expert_mean  # becomes 1
            ax.plot(
                x,
                y,
                label="Expert",
                linewidth=2.5,
                color=COLOR.get("Expert", None),
                linestyle="--",
            )

        for algo_key in algo_keys:
            algo_data = env_data[algo_key]
            if not algo_data["data"]:
                continue

            values = np.vstack(algo_data["data"])
            mean = values.mean(axis=0)
            std = values.std(axis=0)

            # Normalize by expert mean to match your first script behavior
            if normalize_expert and expert_mean is not None and expert_mean != 0:
                mean = mean / expert_mean
                std = std / expert_mean

            color_key = normalize_algo_key(algo_key)
            algo_disp = get_algo_display(algo_key, ALGO_DISPLAY)

            color = COLOR.get(color_key, None)
            linestyle = get_linestyle(algo_disp)

            # Mimic your “FOCUS thicker” logic; adapt as needed
            lw = 1 if algo_key.lower().startswith("focus") else 1

            ax.plot(
                x,
                mean,
                label=algo_disp,
                linewidth=lw,
                color=color,
                linestyle=linestyle,
            )
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=0.2,
                color=color,
            )

        ax.set_title(ENV_NAMES[env_name])
        ax.set_xlabel(x_label)
        ax.set_ylim(bottom=ylim_bottom)
        ax.margins(x=0)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # capture legend from first active axis
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        if not shared_legend:
            ax.legend(frameon=False, fontsize=9)

    # hide unused axes
    for extra_ax in axes[len(env_names):]:
        extra_ax.set_visible(False)

    # shared legend at top, ordered like ALGOS (Expert + your list)
    if shared_legend and legend_handles and legend_labels:
        # reorder according to ALGOS list
        # (labels contain display names, so build desired order in display-space)
        desired = []
        for a in ALGOS:
            desired.append(a)  # ALGOS already includes "Expert"

        # build mapping label -> handle (last one wins, ok)
        map_lh = {lab: h for h, lab in zip(legend_handles, legend_labels)}
        ordered_handles = [map_lh[l] for l in desired if l in map_lh]
        ordered_labels  = [l for l in desired if l in map_lh]

        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            ncol=min(len(ordered_labels), 6),
            frameon=False,
            fontsize=12,
            borderaxespad=0.02,
        )
        fig.tight_layout(rect=[0.01, 0, 1, 0.97])

    fig.supylabel("Normalized Return" if normalize_expert else "Return", x=0.008, fontsize=12)
    # fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_single_env(
    env_name,
    env_data,
    output_path,
    *,
    COLOR,
    ALGOS,
    ALGO_DISPLAY,
    ALGO_INDEX,
    EXPERTS=None,
    ENV_NAMES=None,
    normalize_expert=True,
    add_expert_line=True,
    ylim_bottom=0.0,
):
    """
    Plots a single environment into its own file.
    env_data is data[env_name] with the same structure as in plot_results_like_first.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    # Determine common x
    n_points = None
    max_steps = 0
    for algo_data in env_data.values():
        if algo_data["data"] and n_points is None:
            n_points = algo_data["data"][0].shape[0]
        if algo_data.get("agent_steps"):
            max_steps = max(max_steps, max(algo_data["agent_steps"]))

    if n_points is None:
        plt.close(fig)
        return

    if max_steps > 0:
        x = np.linspace(0, max_steps, num=n_points)
        x_label = "Steps"
    else:
        x = np.linspace(0.0, 1.0, num=n_points)
        x_label = "Training progress"

    # Expert stats
    expert_mean = None
    if EXPERTS and env_name in EXPERTS:
        expert_mean = float(EXPERTS[env_name]["mean"])

    # Optional expert horizontal line
    if add_expert_line and expert_mean is not None:
        y = np.full_like(x, expert_mean, dtype=np.float64)
        if normalize_expert and expert_mean != 0:
            y = y / expert_mean  # -> 1
        ax.plot(
            x,
            y,
            label="Expert",
            linewidth=2.5,
            color=COLOR.get("Expert", None),
            linestyle="--",
        )

    # Sort algos
    algo_keys = sorted(
        env_data.keys(),
        key=lambda k: algo_sort_key(k, ALGO_INDEX, ALGO_DISPLAY),
    )

    for algo_key in algo_keys:
        algo_data = env_data[algo_key]
        if not algo_data["data"]:
            continue

        values = np.vstack(algo_data["data"])
        mean = values.mean(axis=0)
        std = values.std(axis=0)

        if normalize_expert and expert_mean is not None and expert_mean != 0:
            mean = mean / expert_mean
            std = std / expert_mean

        color_key = normalize_algo_key(algo_key)
        algo_disp = get_algo_display(algo_key, ALGO_DISPLAY)

        ax.plot(
            x,
            mean,
            label=algo_disp,
            linewidth=1,  # adjust if you want FOCUS thicker
            color=COLOR.get(color_key, None),
            linestyle=get_linestyle(algo_disp),
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2,
            color=COLOR.get(color_key, None),
        )

    title = ENV_NAMES.get(env_name, env_name) if ENV_NAMES else env_name
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized Return" if normalize_expert else "Return")
    ax.set_ylim(bottom=ylim_bottom)
    ax.margins(x=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Legend in desired order (Expert + ALGOS)
    handles, labels = ax.get_legend_handles_labels()
    map_lh = {lab: h for h, lab in zip(handles, labels)}

    desired = list(ALGOS)
    ordered_handles = [map_lh[l] for l in desired if l in map_lh]
    ordered_labels  = [l for l in desired if l in map_lh]

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=min(len(ordered_labels), 4),
        frameon=False,
        fontsize=11,
        borderaxespad=0.01,
    )
    fig.tight_layout(rect=[0.01, 0, 1, 0.97])


    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def plot_median_across_envs(
    data,
    output_path,
    COLOR,
    ALGOS,
    ALGO_DISPLAY,
    EXPERTS=None,
    *,
    normalize_expert=True,
):
    """
    Builds a single curve per algo = median across environments of the per-env mean curves.
    Assumes all curves already interpolated to same n_points in your data collection step.
    """
    env_names = list(data.keys())
    if not env_names:
        return

    # Determine n_points from first available run
    n_points = None
    for env in env_names:
        for algo_data in data[env].values():
            if algo_data["data"]:
                n_points = algo_data["data"][0].shape[0]
                break
        if n_points is not None:
            break
    if n_points is None:
        return

    # Use training progress (%) on x like your first median plot
    x = np.linspace(0, 100, n_points)

    # collect per-env mean curves
    per_algo_env_means = {a: [] for a in ALGOS if a != "Expert"}  # we draw Expert separately (flat 1)
    for env in env_names:
        expert_mean = None
        if EXPERTS and env in EXPERTS:
            expert_mean = float(EXPERTS[env]["mean"])

        # compute per-algo mean curve for this env
        for algo_key, algo_data in data[env].items():
            if not algo_data["data"]:
                continue
            disp = get_algo_display(algo_key, ALGO_DISPLAY)
            if disp not in per_algo_env_means:
                continue
            values = np.vstack(algo_data["data"])
            mean = values.mean(axis=0)

            if normalize_expert and expert_mean is not None and expert_mean != 0:
                mean = mean / expert_mean

            per_algo_env_means[disp].append(mean)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    # Expert median line is 1 if normalized, otherwise not meaningful across envs; use 1.
    if normalize_expert:
        ax.plot(
            x,
            np.ones_like(x),
            label="Expert",
            linewidth=2.5,
            color=COLOR.get("Expert", None),
            linestyle="--",
        )

    for algo in ALGOS:
        if algo == "Expert":
            continue
        curves = per_algo_env_means.get(algo, [])
        if not curves:
            continue
        stacked = np.vstack(curves)
        median_curve = np.median(stacked, axis=0)

        lw = 3 if algo.lower().startswith("focus") else 2
        color_key = normalize_algo_key(algo)
        ax.plot(
            x,
            median_curve,
            label=algo,
            linewidth=lw,
            color=COLOR.get(color_key, None),
            linestyle="-",
        )

    ax.set_ylim(bottom=0.0)
    ax.margins(x=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Training Progress (%)")
    ax.set_ylabel("Median Normalized Return" if normalize_expert else "Median Return")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def main(
    sync_remote=False,
    update_group_runs=False,
):
    plotter_dir = Path(__file__).resolve().parent
    grouped_exp_path = plotter_dir / "grouped_exp_urls.yaml"
    grouped_runs_path = plotter_dir / "grouped_exp_logdirs.yaml"
    grouped_exp_urls = load_yaml(grouped_exp_path) or {}

    group_runs = None
    if update_group_runs:
        group_runs = {}
        for env_name, algo_urls in grouped_exp_urls.items():
            if not algo_urls:
                continue
            group_runs[env_name] = {}
            for algo_name, run_url in algo_urls.items():
                if not run_url:
                    continue
                group_runs[env_name][algo_name] = []
                runs = get_group_runs(run_url)
                for run in runs:
                    logdir = run.get("logdir")
                    if not logdir:
                        continue
                    group_runs[env_name][algo_name].append(
                        {
                            "logdir": logdir,
                            "seed": run["config"].get("seed"),
                            "run_id": run["run_id"],
                        }
                    )
        save_yaml(grouped_runs_path, group_runs)
    else:
        if grouped_runs_path.is_file():
            group_runs = load_yaml(grouped_runs_path)


    data = {}
    for env_name, algos in group_runs.items():
        data[env_name] = {}
        for algo, algo_values in algos.items():
            data[env_name][algo] = {
                "data": [],
                "agent_steps": [],
                "epochs": []
            }
            for exp in algo_values:
                local_dir = Path(exp["logdir"])
                if sync_remote:
                    # sync exp from the remote server
                    local_dir.mkdir(parents=True, exist_ok=True)
                    sync_exp_from_remote(
                        remote="bao",
                        remote_exp_dir=f"/home/users/c/candidor/models/mineral/{exp['logdir']}/.",
                        local_exp_dir=str(local_dir),
                    )
                
                ep_rew = np.load(local_dir / "ep_rewards_hist.npy")



                try:
                    data_json = load_json(local_dir / "scores.json")
                    agent_steps = data_json["agent_steps"]
                    epoch = data_json["epoch"]
                except:
                    if algo.startswith("FOCUS"):
                        agent_steps = 10000384
                        epoch = 4882

                def moving_average(x, window):
                    x = np.asarray(x, dtype=np.float64)
                    if window <= 1:
                        return x
                    window = min(window, x.size)
                    pad = window // 2

                    x_pad = np.pad(x, (pad, pad), mode="reflect")
                    kernel = np.ones(window, dtype=np.float64) / window
                    y = np.convolve(x_pad, kernel, mode="valid")
                    return y

                n_points = 5000

                # Interpolate to common resolution
                old_x = np.linspace(0.0, 1.0, num=ep_rew.shape[0])
                new_x = np.linspace(0.0, 1.0, num=n_points)
                ep_rew_inter = np.interp(new_x, old_x, ep_rew)

                # Smooth AFTER interpolation
                smooth_window = 50  # typical values: 15–50
                ep_rew_inter = moving_average(ep_rew_inter, smooth_window)


                data[env_name][algo]["data"].append(ep_rew_inter)
                data[env_name][algo]["agent_steps"].append(agent_steps)
                data[env_name][algo]["epochs"].append(epoch)
                print(env_name, algo, ep_rew.shape, agent_steps, epoch)

    plot_path = plotter_dir / "plots" / "ep_rewards_like_first.png"
    plot_results_like_first(
        data,
        plot_path,
        COLOR=COLOR,
        ALGOS=ALGOS,
        ALGO_DISPLAY=ALGO_DISPLAY,
        ALGO_INDEX=ALGO_INDEX,
        EXPERTS=EXPERTS,
        normalize_expert=True,
        add_expert_line=True,
        shared_legend=True,
    )

    # One plot per environment
    per_env_dir = plotter_dir / "plots" / "per_env"
    for env_name, env_data in data.items():
        out_path = per_env_dir / f"{env_name}_ep_rewards.png"
        plot_single_env(
            env_name,
            env_data,
            out_path,
            COLOR=COLOR,
            ALGOS=ALGOS,
            ALGO_DISPLAY=ALGO_DISPLAY,
            ALGO_INDEX=ALGO_INDEX,
            EXPERTS=EXPERTS,
            ENV_NAMES=ENV_NAMES,
            normalize_expert=True,
            add_expert_line=True,
        )


    median_path = plotter_dir / "plots" / "median_rewards_like_first.png"
    plot_median_across_envs(
        data,
        median_path,
        COLOR=COLOR,
        ALGOS=ALGOS,
        ALGO_DISPLAY=ALGO_DISPLAY,
        EXPERTS=EXPERTS,
        normalize_expert=True,
    )



if __name__ == "__main__":
    # sync exp from the remote server
    main(
        sync_remote=False,
        update_group_runs=False,
    )
