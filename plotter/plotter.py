import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import load_yaml, save_yaml, sync_exp_from_remote
from wandb_api import get_group_runs, get_rew_steps_times

sns.set_theme()
sns.set(rc={"axes.facecolor": "#f5f5f5"})

FOLDER_TO_SAVE_PLOTS = Path(__file__).resolve().parent / "plots_ZOCUS"


EXPERTS = {
    'hopper': {'mean': 4812.96875, 'std': 7.324751853942871},
    'ant': {'mean': 9329.505859375, 'std': 37.58782196044922},
    'humanoid': {'mean': 8225.2177734375, 'std': 81.0524673461914},
    'snu_humanoid': {'mean': 6498.921875, 'std': 49.13310623168945},
}

MAX_STEPS_PER_ENV = {
    "hopper": 10_000_000,
    "ant": 10_000_000,
    "humanoid": 15_000_000,
    "snu_humanoid": 15_000_000,
}

# Max wall time per env (seconds). Set to None to disable capping.
MAX_TIME_PER_ENV = {
    "hopper": 60 * 60 * 2,
    "ant": 60 * 60 * 3,
    "humanoid": 60 * 60 * 6,
    "snu_humanoid": 60 * 60 * 6,
}

ENV_NAMES = {"hopper": "Hopper", "ant": "Ant", "humanoid": "Humanoid", "snu_humanoid": "SNU Humanoid"}

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
    "ZOCUS-SAC-l2": "#4C00FF",
    "ZOCUS-SAC-OT-l2": "#0066FF",
    "ZOCUS-SAC-OT-cos": "#0084FF",
    "ZOCUS-PPO-l2": "#FF9100",
    "ZOCUS-PPO-OT-l2": "#FF5100",
    "ZOCUS-PPO-OT-cos": "#FF3C00",
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
    "ZOCUS-SAC-l2",
    "ZOCUS-SAC-OT-l2",
    "ZOCUS-SAC-OT-cos",
    "ZOCUS-PPO-l2",
    "ZOCUS-PPO-OT-l2",
    "ZOCUS-PPO-OT-cos",
]

ALGO_DISPLAY = {}

ALGO_INDEX = {algo: idx for idx, algo in enumerate(ALGOS)}


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


def get_time_axis_max(env_data, *, max_time_cap=None):
    max_time = 0.0
    for algo_name, algo_data in env_data.items():
        for times in algo_data.get("agent_times", []):
            max_time = max(max_time, max(times))

    if max_time_cap is not None and max_time_cap > 0:
        max_time = min(max_time, max_time_cap)

    print("Max time across all algos:", max_time)
    return max_time


def convert_seconds_to_hours(seconds):
    return seconds / 3600.0


def normalize_center_stat(center_stat):
    center_stat = str(center_stat).lower()
    if center_stat == "media":
        center_stat = "median"
    if center_stat not in {"mean", "median"}:
        raise ValueError(f"Unknown center_stat: {center_stat}")
    return center_stat


def normalize_band(band):
    band = str(band).lower()
    if band in {"95_ci", "95%ci", "ci95"}:
        band = "95ci"
    if band not in {"std", "95ci"}:
        raise ValueError(f"Unknown band: {band}")
    return band


def compute_center_and_band(values, *, center_stat="mean", band="std"):
    values = np.asarray(values, dtype=np.float64)
    center_stat = normalize_center_stat(center_stat)
    band = normalize_band(band)

    if center_stat == "mean":
        center = values.mean(axis=0)
    else:
        center = np.median(values, axis=0)

    if band == "std":
        spread = values.std(axis=0)
        lower = center - spread
        upper = center + spread
        return center, lower, upper

    n_runs = values.shape[0]
    if center_stat == "mean":
        if n_runs < 2:
            lower = center.copy()
            upper = center.copy()
        else:
            std = values.std(axis=0, ddof=1)
            sem = std / np.sqrt(n_runs)
            half_width = 1.96 * sem
            lower = center - half_width
            upper = center + half_width
    else:
        lower = np.percentile(values, 2.5, axis=0)
        upper = np.percentile(values, 97.5, axis=0)

    return center, lower, upper


def build_results_table(
    data,
    *,
    EXPERTS=None,
    ALGO_DISPLAY=None,
    ALGO_INDEX=None,
    normalize_expert=True,
    x_axis="steps",
    center_stat="mean",
    band="std",
):
    """Summarize the final comparable return for every environment and algorithm."""
    center_stat = normalize_center_stat(center_stat)
    band = normalize_band(band)
    ALGO_DISPLAY = ALGO_DISPLAY or {}
    ALGO_INDEX = ALGO_INDEX or {}
    rows = []

    for env_name, env_data in data.items():
        expert_mean = None
        if EXPERTS and env_name in EXPERTS:
            expert_mean = float(EXPERTS[env_name]["mean"])

        algo_keys = sorted(
            env_data,
            key=lambda key: algo_sort_key(key, ALGO_INDEX, ALGO_DISPLAY),
        )
        for algo_key in algo_keys:
            algo_data = env_data[algo_key]
            curves = algo_data.get("data", [])
            if not curves:
                continue

            if x_axis == "time":
                timed_curves = []
                for times, returns in zip(algo_data.get("agent_times", []), curves):
                    times = np.asarray(times, dtype=np.float64)
                    returns = np.asarray(returns, dtype=np.float64)
                    if times.size == 0 or returns.size == 0 or times.size != returns.size:
                        continue
                    timed_curves.append((times, returns))
                if not timed_curves:
                    continue

                common_end_time = min(times[-1] for times, _ in timed_curves)
                if common_end_time <= 0:
                    continue
                n_points = min(returns.size for _, returns in timed_curves)
                common_grid = np.linspace(0, common_end_time, num=n_points)
                values = np.vstack([np.interp(common_grid, times, returns) for times, returns in timed_curves])
                final_x = convert_seconds_to_hours(common_end_time)
                final_x_unit = "hours"
            elif x_axis == "steps":
                values = np.vstack(curves)
                step_grids = algo_data.get("agent_steps", [])
                valid_steps = [
                    np.asarray(steps, dtype=np.float64) for steps in step_grids if steps is not None and len(steps) > 0
                ]
                final_x = min(steps[-1] for steps in valid_steps) if valid_steps else np.nan
                final_x_unit = "steps"
            else:
                raise ValueError(f"Unknown x_axis: {x_axis}")

            if normalize_expert and expert_mean is not None and expert_mean != 0:
                values = values / expert_mean

            center, lower, upper = compute_center_and_band(
                values[:, -1:],
                center_stat=center_stat,
                band=band,
            )
            rows.append(
                {
                    "environment": env_name,
                    "environment_name": ENV_NAMES.get(env_name, env_name),
                    "algorithm": algo_key,
                    "algorithm_display": get_algo_display(algo_key, ALGO_DISPLAY),
                    "center": float(center[0]),
                    "lower": float(lower[0]),
                    "upper": float(upper[0]),
                    "n_runs": int(values.shape[0]),
                    "final_x": float(final_x),
                    "final_x_unit": final_x_unit,
                    "x_axis": x_axis,
                    "center_stat": center_stat,
                    "band": band,
                    "normalized_by_expert": normalize_expert,
                }
            )

    return rows


def _latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def save_results_table(
    data,
    output_stem,
    *,
    EXPERTS=None,
    ALGO_DISPLAY=None,
    ALGO_INDEX=None,
    normalize_expert=True,
    x_axis="steps",
    center_stat="mean",
    band="std",
):
    """Save final-return summaries as numeric CSV and a wide LaTeX table."""
    rows = build_results_table(
        data,
        EXPERTS=EXPERTS,
        ALGO_DISPLAY=ALGO_DISPLAY,
        ALGO_INDEX=ALGO_INDEX,
        normalize_expert=normalize_expert,
        x_axis=x_axis,
        center_stat=center_stat,
        band=band,
    )
    if not rows:
        return None

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    tex_path = output_stem.with_suffix(".tex")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    env_names = list(dict.fromkeys(row["environment_name"] for row in rows))
    algo_names = list(dict.fromkeys(row["algorithm_display"] for row in rows))
    values_by_env_algo = {(row["environment_name"], row["algorithm_display"]): row for row in rows}

    def format_cell(row):
        if row is None:
            return "--"
        if band == "std":
            spread = max(row["center"] - row["lower"], row["upper"] - row["center"])
            return rf"{row['center']:.3f} $\pm$ {spread:.3f}"
        return rf"{row['center']:.3f} [{row['lower']:.3f}, {row['upper']:.3f}]"

    latex_lines = [
        rf"\begin{{tabular}}{{l{'c' * len(algo_names)}}}",
        r"\toprule",
        "Environment & " + " & ".join(_latex_escape(name) for name in algo_names) + r" \\",
        r"\midrule",
    ]
    for env_name in env_names:
        cells = [format_cell(values_by_env_algo.get((env_name, algo_name))) for algo_name in algo_names]
        latex_lines.append(_latex_escape(env_name) + " & " + " & ".join(cells) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    return csv_path, tex_path


def plot_results_like_first(
    data,
    output_path,
    COLOR,
    ALGOS,
    ALGO_DISPLAY,
    ALGO_INDEX,
    EXPERTS=None,
    *,
    normalize_expert=True,  # divide by expert mean per env
    add_expert_line=True,  # plot dashed expert line from EXPERTS
    shared_legend=True,  # one legend for the whole figure
    n_cols=4,
    ylim_bottom=0.0,
    ylim_top=1.1,
    x_axis="steps",
    center_stat="mean",
    band="std",
):
    """data[env][algo] = {
        "data": [np.array(n_points), ...]  # per-seed
        "agent_steps": [...],
        "agent_times": [...],
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

        if env_name == "hopper":
            center_stat_ = "mean"
            band_ = "std"
        else:
            center_stat_ = center_stat
            band_ = band

        # Determine common x
        n_points = None
        max_steps = MAX_STEPS_PER_ENV.get(env_name)
        for algo_data in env_data.values():
            if algo_data["data"] and n_points is None:
                n_points = algo_data["data"][0].shape[0]
            if algo_data.get("agent_steps"):
                for steps in algo_data["agent_steps"]:
                    if steps is None or len(steps) == 0:
                        continue
                    max_steps = max(max_steps or 0, float(np.max(steps)))

        if n_points is None:
            ax.set_visible(False)
            continue

        x = None
        x_label = None
        if x_axis == "time":
            max_time = get_time_axis_max(env_data, max_time_cap=MAX_TIME_PER_ENV.get(env_name))
            max_time = convert_seconds_to_hours(max_time)
            x = np.linspace(0, max_time, num=n_points)
            x_label = "Relative Time (h)"
        if x is None:
            if max_steps is not None and max_steps > 0:
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
        print(env_data.keys())
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
                linewidth=1,
                color=COLOR.get("Expert", None),
                linestyle="--",
            )

        for algo_key in algo_keys:
            algo_data = env_data[algo_key]
            if not algo_data["data"]:
                continue

            if x_axis == "time":
                agent_times = algo_data.get("agent_times", [])
                if not agent_times:
                    continue
                agent_times = [convert_seconds_to_hours(t) for t in agent_times]
                min_times_x = min(t[-1] for t in agent_times if len(t) > 0)
                if min_times_x <= 0:
                    continue
                min_times_x_grid = np.linspace(0, min_times_x, num=n_points)
                values = []
                for i in range(len(algo_data["data"])):
                    times = agent_times[i]
                    returns = algo_data["data"][i]
                    if len(times) == 0:
                        continue
                    ep_ret_inter = np.interp(min_times_x_grid, times, returns)
                    values.append(ep_ret_inter)
                if not values:
                    continue
                values = np.vstack(values)
                mean, band_low, band_high = compute_center_and_band(
                    values,
                    center_stat=center_stat_,
                    band=band_,
                )
                x_plot = min_times_x_grid
            else:
                values = np.vstack(algo_data["data"])
                mean, band_low, band_high = compute_center_and_band(
                    values,
                    center_stat=center_stat_,
                    band=band_,
                )
                x_plot = x

            # Normalize by expert mean to match your first script behavior
            if normalize_expert and expert_mean is not None and expert_mean != 0:
                mean = mean / expert_mean
                band_low = band_low / expert_mean
                band_high = band_high / expert_mean

            color_key = normalize_algo_key(algo_key)
            algo_disp = get_algo_display(algo_key, ALGO_DISPLAY)

            color = COLOR.get(color_key, None)
            linestyle = get_linestyle(algo_disp)

            # Mimic your “FOCUS thicker” logic; adapt as needed
            lw = 1 if algo_key.lower().startswith("focus") else 1

            ax.plot(
                x_plot,
                mean,
                label=algo_disp,
                linewidth=lw,
                color=color,
                linestyle=linestyle,
            )
            ax.fill_between(
                x_plot,
                band_low,
                band_high,
                alpha=0.2,
                color=color,
            )

        ax.set_title(ENV_NAMES[env_name])
        ax.set_xlabel(x_label)
        if ylim_top is None:
            ax.set_ylim(bottom=ylim_bottom)
        else:
            ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
        ax.margins(x=0)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # capture legend from first active axis
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        if not shared_legend:
            ax.legend(frameon=False, fontsize=9)

    # hide unused axes
    for extra_ax in axes[len(env_names) :]:
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
        ordered_labels = [l for l in desired if l in map_lh]
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            ncol=len(ordered_labels),
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
    ylim_top=1.1,
    x_axis="steps",
    center_stat="mean",
    band="std",
):
    """Plots a single environment into its own file.
    env_data is data[env_name] with the same structure as in plot_results_like_first.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    # Determine common x
    n_points = None
    max_steps = MAX_STEPS_PER_ENV.get(env_name)
    for algo_data in env_data.values():
        if algo_data["data"] and n_points is None:
            n_points = algo_data["data"][0].shape[0]
        if algo_data.get("agent_steps"):
            for steps in algo_data["agent_steps"]:
                if steps is None or len(steps) == 0:
                    continue
                max_steps = max(max_steps or 0, float(np.max(steps)))

    if n_points is None:
        plt.close(fig)
        return

    x = None
    x_label = None
    if x_axis == "time":
        max_time = get_time_axis_max(env_data, max_time_cap=MAX_TIME_PER_ENV.get(env_name))
        if max_time is not None and max_time > 0:
            max_time = convert_seconds_to_hours(max_time)
            x = np.linspace(0, max_time, num=n_points)
            x_label = "Relative Time (h)"
    if x is None:
        if max_steps is not None and max_steps > 0:
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
            linewidth=1,
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

        if x_axis == "time":
            agent_times = algo_data.get("agent_times", [])
            if not agent_times:
                continue
            agent_times = [convert_seconds_to_hours(t) for t in agent_times]
            min_times_x = min(t[-1] for t in agent_times if len(t) > 0)
            if min_times_x <= 0:
                continue
            min_times_x_grid = np.linspace(0, min_times_x, num=n_points)
            values = []
            for i in range(len(algo_data["data"])):
                times = agent_times[i]
                returns = algo_data["data"][i]
                if len(times) == 0:
                    continue
                ep_ret_inter = np.interp(min_times_x_grid, times, returns)
                values.append(ep_ret_inter)
            if not values:
                continue
            values = np.vstack(values)
            mean, band_low, band_high = compute_center_and_band(
                values,
                center_stat=center_stat,
                band=band,
            )
            x_plot = min_times_x_grid
        else:
            values = np.vstack(algo_data["data"])
            mean, band_low, band_high = compute_center_and_band(
                values,
                center_stat=center_stat,
                band=band,
            )
            x_plot = x

        if normalize_expert and expert_mean is not None and expert_mean != 0:
            mean = mean / expert_mean
            band_low = band_low / expert_mean
            band_high = band_high / expert_mean

        color_key = normalize_algo_key(algo_key)
        algo_disp = get_algo_display(algo_key, ALGO_DISPLAY)

        ax.plot(
            x_plot,
            mean,
            label=algo_disp,
            linewidth=1,  # adjust if you want FOCUS thicker
            color=COLOR.get(color_key, None),
            linestyle=get_linestyle(algo_disp),
        )
        ax.fill_between(
            x_plot,
            band_low,
            band_high,
            alpha=0.2,
            color=COLOR.get(color_key, None),
        )

    title = ENV_NAMES.get(env_name, env_name) if ENV_NAMES else env_name
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized Return" if normalize_expert else "Return")
    if ylim_top is None:
        ax.set_ylim(bottom=ylim_bottom)
    else:
        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    ax.margins(x=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Legend in desired order (Expert + ALGOS)
    handles, labels = ax.get_legend_handles_labels()
    map_lh = {lab: h for h, lab in zip(handles, labels)}

    desired = list(ALGOS)
    ordered_handles = [map_lh[l] for l in desired if l in map_lh]
    ordered_labels = [l for l in desired if l in map_lh]

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=len(ordered_labels),
        frameon=False,
        fontsize=5.5,
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
    center_stat="mean",
    ylim_top=1.1,
):
    """Builds a single curve per algo = median across environments of the per-env mean curves.
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
    center_stat = normalize_center_stat(center_stat)
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
            if center_stat == "mean":
                mean = values.mean(axis=0)
            else:
                mean = np.median(values, axis=0)

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
            linewidth=1,
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

    if ylim_top is None:
        ax.set_ylim(bottom=0.0)
    else:
        ax.set_ylim(bottom=0.0, top=ylim_top)
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
    create_plots=False,
    x_axis="time",
    center_stat="mean",
    band="std",
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
                            "entity": run["entity"],
                            "project": run["project"],
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
                "agent_times": [],
            }
            for exp in algo_values:
                local_dir = Path(exp["logdir"])
                if sync_remote and not local_dir.exists():
                    # sync exp from the remote server
                    local_dir.mkdir(parents=True, exist_ok=True)
                    sync_exp_from_remote(
                        remote="bao",
                        remote_exp_dir=f"/home/users/c/candidor/models/mineral/{exp['logdir']}/.",
                        local_exp_dir=str(local_dir),
                    )

                if create_plots:
                    need_tb_reload = False
                    try:
                        ep_rew = np.load(local_dir / "my_ep_rewards_hist.npy")
                        steps = np.load(local_dir / "my_ep_steps_hist.npy")
                        times = np.load(local_dir / "my_ep_times_hist.npy")
                        if x_axis == "time":
                            if (
                                ep_rew.size == 0
                                or steps.size == 0
                                or times.size == 0
                                or ep_rew.size != steps.size
                                or times.size != steps.size
                            ):
                                need_tb_reload = True
                    except Exception:
                        need_tb_reload = True

                    if need_tb_reload:
                        try:
                            # load tb file
                            tb_dir = local_dir / "tb"

                            if not tb_dir.exists():
                                raise FileNotFoundError(f"TB directory not found: {tb_dir}")

                            # Load everything (scalars only is cheap)
                            ea = EventAccumulator(
                                str(tb_dir),
                                size_guidance={"scalars": 0},
                            )
                            ea.Reload()

                            tag = "train_scores/episode_rewards"
                            if tag not in ea.Tags().get("scalars", []):
                                raise KeyError(f"Scalar '{tag}' not found in {tb_dir}")

                            events = ea.Scalars(tag)
                            steps = np.array([e.step for e in events], dtype=np.int64)
                            ep_rew = np.array([e.value for e in events], dtype=np.float64)
                            times = np.array([e.wall_time for e in events], dtype=np.float64)

                            # plt.plot(steps, ep_rew)
                            # plt.title(f"Loaded from npy: {local_dir}")
                            # plt.show()
                            # plt.close()

                            print(ep_rew[:10])
                            print(steps[:10])
                            print(len(steps), len(ep_rew))

                        except:
                            print("Failed to load TB data from npy, reloading from wandb")
                            ep_rew, steps, times = get_rew_steps_times(
                                entity=exp["entity"],
                                project=exp["project"],
                                run_id=exp["run_id"],
                            )

                        np.save(local_dir / "my_ep_rewards_hist.npy", ep_rew)
                        np.save(local_dir / "my_ep_steps_hist.npy", steps)
                        np.save(local_dir / "my_ep_times_hist.npy", times)

                    print(ep_rew[:10])
                    print(steps[:10])
                    print(times[:10])
                    print(len(steps), len(ep_rew), len(times))

                    max_steps = None if x_axis == "time" else MAX_STEPS_PER_ENV.get(env_name)
                    max_time_env = MAX_TIME_PER_ENV.get(env_name) if x_axis == "time" else None
                    if max_steps is not None:
                        keep_mask = steps <= max_steps
                        steps = steps[keep_mask]
                        ep_rew = ep_rew[keep_mask]

                    if steps.size == 0:
                        continue

                    print(steps[-1])

                    def moving_average(x, window):
                        x = np.asarray(x, dtype=np.float64)
                        if window <= 1:
                            return x
                        window = min(window, x.size)
                        pad = window // 2

                        x_pad = np.pad(x, (pad, pad), mode="reflect")
                        kernel = np.ones(window, dtype=np.float64) / window
                        y = np.convolve(x_pad, kernel, mode="valid")
                        return y[:-1]

                    n_points = 1000
                    if x_axis == "time":
                        print(f"Using wall time for {env_name} {algo}")
                        rel_times = times - times[0]
                        if max_time_env is not None:
                            keep_mask = rel_times <= max_time_env
                            rel_times = rel_times[keep_mask]
                            ep_rew = ep_rew[keep_mask]
                            steps = steps[keep_mask]
                            if rel_times.size == 0:
                                continue
                        print(rel_times[:10])
                        max_time = float(rel_times[-1])
                        time_grid = np.linspace(0, max_time, num=n_points)
                        ep_rew_inter = np.interp(time_grid, rel_times, ep_rew)
                        step_grid = None
                    elif x_axis == "steps":
                        # Interpolate to a shared step grid
                        time_grid = None
                        if max_steps is None:
                            max_steps = float(steps[-1])
                        step_grid = np.linspace(0, max_steps, num=n_points)
                        ep_rew_inter = np.interp(step_grid, steps, ep_rew)
                    else:
                        raise ValueError(f"Unknown x_axis: {x_axis}")

                    # Smooth AFTER interpolation
                    smooth_window = 50  # typical values: 15–50
                    ep_rew_inter = moving_average(ep_rew_inter, smooth_window)

                    data[env_name][algo]["data"].append(ep_rew_inter)
                    if step_grid is not None:
                        data[env_name][algo]["agent_steps"].append(step_grid)
                    if time_grid is not None:
                        data[env_name][algo]["agent_times"].append(time_grid)
                    if x_axis == "time" and step_grid is None:
                        last_x = max_time
                    else:
                        last_x = step_grid[-1]
                    print(env_name, algo, ep_rew.shape, last_x)

    if create_plots:
        table_stem = FOLDER_TO_SAVE_PLOTS / f"final_return_table_{center_stat}{band}_{x_axis}"
        save_results_table(
            data,
            table_stem,
            EXPERTS=EXPERTS,
            ALGO_DISPLAY=ALGO_DISPLAY,
            ALGO_INDEX=ALGO_INDEX,
            normalize_expert=True,
            x_axis=x_axis,
            center_stat=center_stat,
            band=band,
        )

        plot_path = FOLDER_TO_SAVE_PLOTS / f"all_return_{center_stat}{band}_{x_axis}.png"
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
            x_axis=x_axis,
            center_stat=center_stat,
            band=band,
        )

        # One plot per environment
        per_env_dir = FOLDER_TO_SAVE_PLOTS / "per_env"
        for env_name, env_data in data.items():
            out_path = per_env_dir / f"{center_stat}{band}_{env_name}_{x_axis}.png"
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
                x_axis=x_axis,
                center_stat=center_stat,
                band=band,
            )

        median_path = FOLDER_TO_SAVE_PLOTS / "median_rewards.png"
        plot_median_across_envs(
            data,
            median_path,
            COLOR=COLOR,
            ALGOS=ALGOS,
            ALGO_DISPLAY=ALGO_DISPLAY,
            EXPERTS=EXPERTS,
            normalize_expert=True,
            center_stat=center_stat,
        )


if __name__ == "__main__":
    # sync exp from the remote server
    for center_stat in ["mean", "median"]:
        for band in ["std", "95ci"]:
            for x_axis in ["time", "steps"]:
                main(
                    sync_remote=False,
                    update_group_runs=False,
                    create_plots=True,
                    x_axis=x_axis,
                    center_stat=center_stat,
                    band=band,
                )

    # main(
    #     sync_remote=True,
    #     update_group_runs=True,
    #     create_plots=True,
    #     x_axis="time",
    #     center_stat="mean",
    #     band="std",
    # )
