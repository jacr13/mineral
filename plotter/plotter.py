import csv
import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import trim_mean
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import load_yaml, save_yaml, sync_exp_from_remote
from wandb_api import get_group_runs, get_rew_steps_times

sns.set_theme()
sns.set(rc={"axes.facecolor": "#f5f5f5"})

FOLDER_TO_SAVE_PLOTS = Path(__file__).resolve().parent / "plots8"

# Moving-average window applied to every curve after interpolation (typical values: 15-50).
SMOOTH_WINDOW = 50


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
    # Same hues as before, dialed from near-primary saturation down to
    # mid-tone so they don't out-shout FOCUS/ILD's softer palette; re-validated
    # (adjacent-pair CVD/chroma/normal-vision all pass).
    "SAMfO/DACfO": "#3E7AE0",
    "OPOLO": "#C5263D",
    "GAIfO": "#ED8326",
    "MAAD": "#DF3AA5",
    "PWIL": "#A6780E",
    "OOPS": "#9448C9",
    "LWAIL": "#2B4FA0",
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
    "PWIL",
    "OOPS",
    "LWAIL",
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

IGNORE_ALGOS = [
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


# FOCUS variants and ILD stay solid (the figure's main track); the other six
# imitation-learning baselines each get a distinct dash pattern since their
# colors alone (e.g. OPOLO/MAAD, both pink-magenta hues) can be hard to tell
# apart where curves and bands overlap.
LINESTYLE = {
    "Expert": "--",
    "SAMfO/DACfO": "-.",
    "OPOLO": ":",
    "GAIfO": (0, (3, 1, 1, 1)),
    "MAAD": (0, (5, 2)),
    "PWIL": (0, (1, 1)),
    "OOPS": (0, (3, 3, 1, 3)),
    "LWAIL": (0, (4, 1, 1, 1, 1, 1)),
    "ILD": "-",
    "FOCUS-l2": "-",
    "FOCUS-OT-l2": "-",
    "FOCUS-OT-cos": "-",
}


def LIGHT_IQM_INCLUDE(algo):
    """Algorithms shown in the lighter aggregate IQM figure.

    OOPS is left out (pooled IQM ~0.01 on both axes); PWIL is the strongest remaining baseline on wall-time
    (~0.40) and has a color that's easy to tell apart from LWAIL's (though it drops to ~0.10 on steps).
    """
    return algo.startswith("FOCUS") or algo in {"ILD", "LWAIL", "PWIL"}


def get_linestyle(algo_disp):
    return LINESTYLE.get(algo_disp, "-")


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
    if center_stat not in {"mean", "median", "iqm"}:
        raise ValueError(f"Unknown center_stat: {center_stat}")
    return center_stat


def normalize_band(band):
    band = str(band).lower()
    if band in {"95_ci", "95%ci", "ci95"}:
        band = "95ci"
    if band not in {"std", "95ci"}:
        raise ValueError(f"Unknown band: {band}")
    return band


def interquartile_mean(values, axis=0):
    """Compute a 25% trimmed mean, trimming floor(n / 4) from each tail.

    With fewer than four runs, no observations are trimmed.
    """
    return trim_mean(values, proportiontocut=0.25, axis=axis)


def get_center_statistic(center_stat):
    return {"mean": np.mean, "median": np.median, "iqm": interquartile_mean}[normalize_center_stat(center_stat)]


def compute_center_and_band(values, *, center_stat="mean", band="std"):
    """Return a standard-deviation band or a percentile bootstrap 95% CI.

    Bootstrap intervals use 10,000 resamples of runs and a fixed seed.
    They are pointwise intervals for the selected mean, median, or IQM, not a
    simultaneous confidence band for the entire curve.
    """
    values = np.asarray(values, dtype=np.float64)
    center_stat = normalize_center_stat(center_stat)
    band = normalize_band(band)

    statistic = get_center_statistic(center_stat)
    center = statistic(values, axis=0)

    if band == "std":
        spread = values.std(axis=0)
        lower = center - spread
        upper = center + spread
        return center, lower, upper

    n_runs = values.shape[0]
    if n_runs == 0:
        raise ValueError("Bootstrap confidence intervals require at least one run")
    if n_runs == 1:
        return center, center.copy(), center.copy()

    n_resamples = 10_000
    rng = np.random.default_rng(0)
    indices = rng.integers(n_runs, size=(n_resamples, n_runs))
    flat_values = values.reshape(n_runs, -1)
    bounds = np.empty((2, flat_values.shape[1]))
    # Bound temporary sample arrays and reuse run draws across all points
    # to preserve dependence within each run's trajectory.
    chunk_size = max(1, 1_000_000 // (n_resamples * n_runs))
    for start in range(0, flat_values.shape[1], chunk_size):
        stop = start + chunk_size
        samples = flat_values[:, start:stop][indices]
        estimates = statistic(samples, axis=1)
        bounds[:, start:stop] = np.percentile(estimates, [2.5, 97.5], axis=0)
    lower = bounds[0].reshape(center.shape)
    upper = bounds[1].reshape(center.shape)

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
                max_time_cap = MAX_TIME_PER_ENV.get(env_name)
                if max_time_cap is not None and max_time_cap > 0:
                    common_end_time = min(common_end_time, max_time_cap)
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
                max_steps_cap = MAX_STEPS_PER_ENV.get(env_name)
                if max_steps_cap is not None and not np.isnan(final_x):
                    final_x = min(final_x, max_steps_cap)
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

    band = normalize_band(band)
    present_envs = list(dict.fromkeys(row["environment_name"] for row in rows))
    preferred_envs = ["Hopper", "Ant", "Humanoid", "SNU Humanoid"]
    env_names = [env for env in preferred_envs if env in present_envs]
    env_names.extend(env for env in present_envs if env not in env_names)
    algo_names = list(dict.fromkeys(row["algorithm_display"] for row in rows))
    preferred_algos = [
        "SAMfO/DACfO",
        "OPOLO",
        "GAIfO",
        "MAAD",
        "PWIL",
        "OOPS",
        "LWAIL",
        "ILD",
        "FOCUS-l2",
        "FOCUS-OT-l2",
        "FOCUS-OT-cos",
    ]
    order = {name: index for index, name in enumerate(preferred_algos)}
    algo_names.sort(key=lambda name: (name.startswith("FOCUS"), order.get(name, -1)))
    values_by_env_algo = {(row["environment_name"], row["algorithm_display"]): row for row in rows}
    best_by_env = {
        env: max(
            (
                row["center"]
                for row in rows
                if row["environment_name"] == env and row["algorithm_display"] != "Expert" and np.isfinite(row["center"])
            ),
            default=None,
        )
        for env in env_names
    }
    method_labels = {
        "FOCUS-l2": r"\mytitleshort-L2",
        "FOCUS-OT-l2": r"\mytitleshort-OT-L2",
        "FOCUS-OT-cos": r"\mytitleshort-OT-Cos",
    }

    def format_cell(row, env):
        if row is None:
            return "--"
        if band == "std":
            spread = max(row["center"] - row["lower"], row["upper"] - row["center"])
            content = rf"{row['center']:.3f} \pm {spread:.3f}"
        else:
            # Preserve asymmetric bootstrap intervals exactly.
            content = rf"{row['center']:.3f}\;[{row['lower']:.3f}, {row['upper']:.3f}]"
        if row["algorithm_display"] != "Expert" and row["center"] == best_by_env[env]:
            content = rf"\mathbf{{{content}}}"
        return f"${content}$"

    latex_lines = [
        rf"\begin{{tabular}}{{l{'c' * len(env_names)}}}",
        r"  \toprule",
        "  Method & " + " & ".join(_latex_escape(name) for name in env_names) + r" \\",
        r"  \midrule",
        "",
    ]
    focus_started = False
    for index, algo_name in enumerate(algo_names):
        if algo_name.startswith("FOCUS") and not focus_started:
            if index:
                latex_lines.extend([r"  \midrule", ""])
            focus_started = True
        latex_lines.append("  " + method_labels.get(algo_name, _latex_escape(algo_name)))
        for env_index, env_name in enumerate(env_names):
            cell = format_cell(values_by_env_algo.get((env_name, algo_name)), env_name)
            ending = r" \\" if env_index == len(env_names) - 1 else ""
            latex_lines.append("  & " + cell + ending)
        latex_lines.append("")
    latex_lines.extend([r"  \bottomrule", r"\end{tabular}", ""])
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

    # Wrap the legend into two rows: a single row of Expert + every method is wider than this
    # 6.5" figure and gets clipped on both sides.
    legend_ncol = math.ceil(len(ordered_labels) / 2)
    legend_rows = math.ceil(len(ordered_labels) / legend_ncol)
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        fontsize=6.5,
        borderaxespad=0.01,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0.01, 0, 1, 1 - 0.04 * legend_rows])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def equal_environment_iqm(values, weights):
    """Average the middle 50% of weighted score mass along the penultimate axis.

    Each environment has total mass 1 / n_environments. Observations crossing
    the 25% or 75% boundary contribute only their overlapping mass.
    """
    order = np.argsort(values, axis=-2)
    sorted_values = np.take_along_axis(values, order, axis=-2)
    weights = np.broadcast_to(np.asarray(weights)[:, None], values.shape)
    sorted_weights = np.take_along_axis(weights, order, axis=-2)
    upper_mass = np.cumsum(sorted_weights, axis=-2)
    lower_mass = upper_mass - sorted_weights
    retained = np.maximum(0, np.minimum(upper_mass, 0.75) - np.maximum(lower_mass, 0.25))
    return np.sum(sorted_values * retained, axis=-2) / 0.5


def compute_aggregate_iqm(values, *, n_resamples=10_000, seed=0):
    """Compute equally weighted task IQM with a stratified bootstrap CI.

    Accept one (seed, point) array per environment, with unequal seed counts.
    Resample the original number of seeds independently within each fixed
    environment, preserving whole trajectories across training points.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if not arrays or any(a.ndim != 2 or 0 in a.shape for a in arrays):
        raise ValueError("Expected nonempty (seed, point) arrays per environment")
    if len({a.shape[1] for a in arrays}) != 1:
        raise ValueError("All environments must use the same number of points")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Aggregate scores must be finite")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    n_envs, n_points = len(arrays), arrays[0].shape[1]
    counts = [a.shape[0] for a in arrays]
    weights = np.concatenate([np.full(n, 1 / (n_envs * n)) for n in counts])
    center = equal_environment_iqm(np.concatenate(arrays), weights)
    rng = np.random.default_rng(seed)
    indices = [rng.integers(n, size=(n_resamples, n)) for n in counts]
    bounds = np.empty((2, n_points))
    chunk_size = max(1, 1_000_000 // (n_resamples * sum(counts)))
    for start in range(0, n_points, chunk_size):
        samples = np.concatenate([a[:, start : start + chunk_size][draw] for a, draw in zip(arrays, indices)], axis=1)
        estimates = equal_environment_iqm(samples, weights)
        bounds[:, start : start + chunk_size] = np.percentile(estimates, [2.5, 97.5], axis=0)
    return center, bounds[0], bounds[1]


def build_aggregate_iqm_results(data, *, x_axis, n_points=1000):
    """Normalize and align algorithms represented in every benchmark task."""
    if x_axis not in {"time", "steps"}:
        raise ValueError(f"Unknown x_axis: {x_axis}")
    env_names = list(ENV_NAMES)
    grid_key = "agent_times" if x_axis == "time" else "agent_steps"
    caps = MAX_TIME_PER_ENV if x_axis == "time" else MAX_STEPS_PER_ENV
    prepared = {env: {} for env in env_names}
    for env in env_names:
        for key, entry in data.get(env, {}).items():
            algo = get_algo_display(key, ALGO_DISPLAY)
            if algo not in ALGOS or algo == "Expert":
                continue
            curves = entry.get("data", [])
            grids = entry.get(grid_key, [])
            if len(curves) != len(grids):
                raise ValueError(f"Missing {grid_key} for {env}/{algo}")
            for grid, curve in zip(grids, curves):
                grid, curve = np.asarray(grid), np.asarray(curve)
                if (
                    grid.ndim != 1
                    or curve.ndim != 1
                    or grid.size < 2
                    or grid.size != curve.size
                    or not np.isfinite(grid).all()
                    or not np.isfinite(curve).all()
                    or np.any(np.diff(grid) <= 0)
                ):
                    raise ValueError(f"Invalid aggregate curve for {env}/{algo}")
                prepared[env].setdefault(algo, []).append((grid, curve))

    eligible = []
    for algo in ALGOS:
        counts = [len(prepared[env].get(algo, [])) for env in env_names]
        if not any(counts):
            continue
        if not all(counts):
            warnings.warn(
                f"Skipping aggregate IQM for {algo}: seed counts by environment "
                f"{dict(zip(env_names, counts))}; at least one run per environment required",
                stacklevel=2,
            )
            continue
        eligible.append(algo)
    if not eligible:
        return {}

    aligned = {algo: [] for algo in eligible}
    endpoints = {}
    for env in env_names:
        runs = [run for algo in eligible for run in prepared[env][algo]]
        start = max(grid[0] for grid, _ in runs)
        end = min(grid[-1] for grid, _ in runs)
        if caps.get(env) is not None:
            end = min(end, caps[env])
        if end <= start:
            raise ValueError(f"No shared training interval for {env}")
        expert = float(EXPERTS[env]["mean"])
        if not np.isfinite(expert) or expert <= 0:
            raise ValueError(f"Positive expert return required for {env}")
        grid = np.linspace(start, end, n_points)
        endpoints[env] = {"start": float(start), "end": float(end)}
        for algo in eligible:
            aligned[algo].append(
                np.vstack([np.interp(grid, run_grid, curve) / expert for run_grid, curve in prepared[env][algo]])
            )
    results = {}
    for algo, arrays in aligned.items():
        center, lower, upper = compute_aggregate_iqm(arrays)
        results[algo] = dict(
            center=center,
            lower=lower,
            upper=upper,
            n_envs=len(env_names),
            n_runs_per_env=dict(zip(env_names, [a.shape[0] for a in arrays])),
            endpoints=endpoints,
        )
    return results


def save_aggregate_iqm(data, output_stem, *, x_axis):
    """Save the pooled IQM curve and final score with a stratified 95% CI."""
    results = build_aggregate_iqm_results(data, x_axis=x_axis)
    if not results:
        return
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    legend_rows = math.ceil(len(results) / 3)
    fig = plt.figure(figsize=(8, 3.8 + 0.25 * legend_rows), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[0.25 * legend_rows, 3])
    legend_ax = fig.add_subplot(grid[0])
    legend_ax.set_axis_off()
    ax = fig.add_subplot(grid[1])
    rows = []
    for algo, result in results.items():
        center, lower, upper = (result[key] for key in ("center", "lower", "upper"))
        x = np.linspace(0, 100, len(center))
        (line,) = ax.plot(
            x,
            center,
            label=algo,
            color=COLOR.get(normalize_algo_key(algo)),
            linewidth=1.5 if algo.lower().startswith("focus") else 1.5,
            linestyle=get_linestyle(algo),
        )
        ax.fill_between(x, lower, upper, color=line.get_color(), alpha=0.2)
        rows.append(
            dict(
                algorithm=algo,
                iqm=float(center[-1]),
                lower=float(lower[-1]),
                upper=float(upper[-1]),
                n_environments=result["n_envs"],
                n_runs_per_environment=json.dumps(result["n_runs_per_env"]),
                x_axis=x_axis,
                environment_intervals=json.dumps(result["endpoints"]),
            )
        )
    ax.set_xlabel(f"{'Wall-time' if x_axis == 'time' else 'Step'} progress (%)")
    ax.set_ylabel("IQM Expert-Normalized Return")
    ax.margins(x=0)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xticks(np.arange(0, 101, 20))
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=8)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return results


def save_combined_aggregate_iqm(
    results_by_axis, output_dir, *, orientations=("vertical", "horizontal"), include=None, suffix=""
):
    """Save time/step IQM panels in vertical and horizontal shared-legend layouts.

    `include` optionally restricts the plotted algorithms (a predicate on the algorithm name; each
    algorithm's IQM curve is computed independently, so filtering doesn't change any curve), and
    `suffix` is appended to the output filenames (e.g. "_light").
    """
    axes_order = ("time", "steps")
    if include is not None:
        results_by_axis = {
            axis: {algo: result for algo, result in results_by_axis.get(axis, {}).items() if include(algo)}
            for axis in axes_order
        }
    if any(not results_by_axis.get(axis) for axis in axes_order):
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = list(dict.fromkeys(algo for axis in axes_order for algo in results_by_axis[axis]))
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {algo: COLOR.get(normalize_algo_key(algo), cycle[i % len(cycle)]) for i, algo in enumerate(algorithms)}
    paths = []
    for orientation in orientations:
        horizontal = orientation == "horizontal"
        ncols = len(algorithms) if horizontal else 3
        legend_rows = math.ceil(len(algorithms) / ncols)
        fig = plt.figure(
            figsize=(14 if horizontal else 8, 3.0 if horizontal else 7 + 0.25 * legend_rows),
            layout="constrained",
        )
        grid = fig.add_gridspec(
            2 if horizontal else 3,
            2 if horizontal else 1,
            height_ratios=[0.3 if horizontal else 0.25 * legend_rows] + ([2.2] if horizontal else [3, 3]),
        )
        legend_ax = fig.add_subplot(grid[0, :])
        legend_ax.set_axis_off()
        panels = []
        legend_handles = {}
        for index, axis in enumerate(axes_order):
            ax = fig.add_subplot(grid[1, index] if horizontal else grid[index + 1, 0], sharey=panels[0] if panels else None)
            panels.append(ax)
            for algo, result in results_by_axis[axis].items():
                x = np.linspace(0, 100, len(result["center"]))
                (line,) = ax.plot(
                    x,
                    result["center"],
                    label=algo,
                    color=colors[algo],
                    linewidth=1.5 if algo.lower().startswith("focus") else 1.5,
                )
                ax.fill_between(x, result["lower"], result["upper"], color=colors[algo], alpha=0.2)
                legend_handles.setdefault(algo, line)
            ax.set_xlabel(f"{'Wall-time' if axis == 'time' else 'Step'} progress (%)")
            if not horizontal or index == 0:
                ax.set_ylabel("IQM Expert-Normalized Return")
            else:
                ax.tick_params(labelleft=False)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.margins(x=0)
            ax.spines[["right", "top"]].set_visible(False)
        labels = algorithms
        legend = legend_ax.legend(
            [legend_handles[label] for label in labels],
            labels,
            loc="center",
            ncol=ncols,
            frameon=False,
            fontsize=11 if horizontal else 8,
            columnspacing=1.0,
            handletextpad=0.5,
        )
        if horizontal:
            # Measure before layout so an oversized legend cannot collapse axes.
            fig.set_layout_engine(None)
            fig.canvas.draw()
            legend_width = legend.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
            if legend_width + 0.6 > fig.get_figwidth():
                fig.set_size_inches(legend_width + 0.6, fig.get_figheight())
            fig.set_layout_engine("constrained")
        path = output_dir / f"aggregate_iqm95ci_{orientation}{suffix}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


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
    """Builds a single curve per algo = median across environments of the per-env center curves.
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
            mean = get_center_statistic(center_stat)(values, axis=0)

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

        lw = 1.5 if algo.lower().startswith("focus") else 1.5
        color_key = normalize_algo_key(algo)
        ax.plot(
            x,
            median_curve,
            label=algo,
            linewidth=lw,
            color=COLOR.get(color_key, None),
            linestyle=get_linestyle(algo),
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
    aggregate_only=False,
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
            if algo in IGNORE_ALGOS:
                continue
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
                        if window <= 1 or x.size <= 1:
                            return x
                        window = min(window, x.size)
                        pad = window // 2

                        x_pad = np.pad(x, (pad, pad), mode="reflect")
                        kernel = np.ones(window, dtype=np.float64) / window
                        y = np.convolve(x_pad, kernel, mode="valid")
                        # Even windows produce one extra point; odd windows
                        # already match the input length. Keep the x-grid aligned.
                        return y[:x.size]

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
                    smooth_window = SMOOTH_WINDOW
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

    if create_plots and aggregate_only:
        return build_aggregate_iqm_results(data, x_axis=x_axis)

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

        return save_aggregate_iqm(
            data,
            FOLDER_TO_SAVE_PLOTS / f"aggregate_iqm95ci_{x_axis}",
            x_axis=x_axis,
        )


if __name__ == "__main__":
    # True: only the horizontal time/steps IQM figure. False: all outputs.
    ONLY_HORIZONTAL_AGGREGATED_IQM = False

    aggregate_results = {}
    center_stats = ["mean"] if ONLY_HORIZONTAL_AGGREGATED_IQM else ["mean", "median"]
    for center_stat in center_stats:
        for band in ["95ci"]:  # ["95ci", "std"]:
            for x_axis in ["time", "steps"]:
                aggregate_results[x_axis] = main(
                    sync_remote=False,
                    update_group_runs=False,
                    create_plots=True,
                    x_axis=x_axis,
                    center_stat=center_stat,
                    band=band,
                    aggregate_only=ONLY_HORIZONTAL_AGGREGATED_IQM,
                )

    save_combined_aggregate_iqm(
        aggregate_results,
        FOLDER_TO_SAVE_PLOTS,
        orientations=("horizontal",) if ONLY_HORIZONTAL_AGGREGATED_IQM else ("vertical", "horizontal"),
    )

    # Lighter figure with only the headline methods (FOCUS variants, ILD, LWAIL, PWIL).
    save_combined_aggregate_iqm(
        aggregate_results,
        FOLDER_TO_SAVE_PLOTS,
        orientations=("horizontal",),
        include=LIGHT_IQM_INCLUDE,
        suffix="_light",
    )
