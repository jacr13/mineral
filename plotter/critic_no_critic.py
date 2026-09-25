"""Critic vs no-critic reward-shaping ablation: downloads wandb data, then
produces training-curve plots and a LaTeX/CSV final-return table.

Sweeps live in one wandb project per environment (see PROJECTS below). Every
run in a project carries these config fields under agent.otil:
    critic_disabled          bool  -- True => "no-critic", False => "critic"
    critic_reward_mapping    str   -- "exp" | "log_exp" | "neg"
    critic_reward_shapping   bool  -- reward-shaping on/off (default filter: True)
    loss_ot_cost_type        str   -- "l2" | "cosine"

Each (critic_disabled, critic_reward_mapping, loss_ot_cost_type) triple maps
to one wandb group (~6 seeds) and becomes one "method", e.g. "critic-exp-l2"
or "no-critic-logexp-cos".
"""

import csv
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wandb_api import API, get_rew_steps_times

from plotter import (
    ENV_NAMES,
    EXPERTS,
    MAX_STEPS_PER_ENV,
    MAX_TIME_PER_ENV,
    _latex_escape,
    compute_center_and_band,
    convert_seconds_to_hours,
    get_center_statistic,
    normalize_band,
    normalize_center_stat,
)

sns.set_theme()
sns.set(rc={"axes.facecolor": "#f5f5f5"})

PLOTTER_DIR = Path(__file__).resolve().parent
FOLDER_TO_SAVE_PLOTS = PLOTTER_DIR / "crit_no_critic_smoothed25"
CACHE_DIR = FOLDER_TO_SAVE_PLOTS / "cache"

ENTITY = "jacr"
PROJECTS = {
    "hopper": "new_v2_critic_OTIL_SAPO-dflex_hopper-slurm",
    "ant": "new_v2_critic_OTIL_SAPO-dflex_ant-slurm",
    "humanoid": "new_v2_critic_OTIL_SAPO-dflex_humanoid-slurm",
    "snu_humanoid": "new_v2_critic_OTIL_SAPO-dflex_snu_humanoid-slurm",
}

# Runs superseded by a rerun, excluded so a method isn't averaged over both
# batches. critic-logexp-l2 / Humanoid: the original 6 seeds all landed on a
# Tesla P100 node (gpu006) and were wall-clock-capped at ~11.5M of the 15M
# steps; the rerun on Turing/Ampere GPUs (scripts/otil.sh, GPU_CONSTRAINT)
# reached the full 15M. Remove these entries to pool both batches again.
EXCLUDED_RUN_IDS = {
    "illinois-aspen-kilo-nebraska-london-2UoK.OTIL.DFlex_humanoid_64.seed110",
    "illinois-aspen-kilo-nebraska-london-51Di.OTIL.DFlex_humanoid_64.seed1000",
    "illinois-aspen-kilo-nebraska-london-6SQa.OTIL.DFlex_humanoid_64.seed1200",
    "illinois-aspen-kilo-nebraska-london-97Pl.OTIL.DFlex_humanoid_64.seed100",
    "illinois-aspen-kilo-nebraska-london-Dgbq.OTIL.DFlex_humanoid_64.seed120",
    "illinois-aspen-kilo-nebraska-london-lmlz.OTIL.DFlex_humanoid_64.seed1100",
}

MAPPING_LABEL = {"exp": "exp", "log_exp": "logexp", "neg": "neg"}
COST_LABEL = {"l2": "l2", "cosine": "cos"}

CRITIC_METHODS = [
    "critic-exp-l2",
    "critic-exp-cos",
    "critic-logexp-l2",
    "critic-logexp-cos",
    "critic-neg-l2",
]
NO_CRITIC_METHODS = [f"no-{method}" for method in CRITIC_METHODS]
# Paired (critic-X, no-critic-X) rather than all critics then all no-critics,
# so the table/legend put each ablation pair next to each other for a direct
# comparison instead of two separate blocks far apart.
METHOD_ORDER = [method for pair in zip(CRITIC_METHODS, NO_CRITIC_METHODS) for method in pair]
# Expert goes last: legends fill column by column, so with it first every
# (critic-X, no-critic-X) pair below would be shifted out of its column.
ALGOS = METHOD_ORDER + ["Expert"]
ALGO_INDEX = {algo: idx for idx, algo in enumerate(ALGOS)}

# Categorical hues assigned in fixed order (blue, orange, aqua, yellow,
# magenta -- first five slots of the validated dataviz categorical theme),
# one per reward-mapping/cost-type pair. Critic vs no-critic is redundantly
# coded on top of that: full saturation + solid line for critic, a pale tint
# of the same hue + dashed line for no-critic -- so the two are easy to tell
# apart even where curves and bands overlap, not just by dash pattern alone.
COLOR_BASE = {
    "exp-l2": "#2a78d6",
    "exp-cos": "#eb6834",
    "logexp-l2": "#1baf7a",
    "logexp-cos": "#eda100",
    "neg-l2": "#e87ba4",
}


def _lighten(hex_color, amount=0.5):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


COLOR = {"Expert": "#616161"}
for _method in METHOD_ORDER:
    _base_key = _method.replace("no-critic-", "").replace("critic-", "")
    _hue = COLOR_BASE[_base_key]
    COLOR[_method] = _lighten(_hue, 0.45) if _method.startswith("no-critic") else _hue


def method_name(critic_disabled, reward_mapping, cost_type):
    prefix = "no-critic" if critic_disabled else "critic"
    mapping_label = MAPPING_LABEL.get(reward_mapping, reward_mapping)
    cost_label = COST_LABEL.get(cost_type, cost_type)
    return f"{prefix}-{mapping_label}-{cost_label}"


def get_linestyle(method):
    if method == "Expert":
        return ":"
    return "--" if method.startswith("no-critic") else "-"


def get_linewidth(method):
    if method == "Expert":
        return 1.0
    return 1.7 if method.startswith("no-critic") else 2.1


def moving_average(x, window):
    x = np.asarray(x, dtype=np.float64)
    if window <= 1:
        return x
    window = min(window, x.size)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x_pad, kernel, mode="valid")


def fetch_run_history(project, run_id, *, force_refresh=False):
    """Download (and cache locally) episode-reward/step/time history for one run."""
    run_cache_dir = CACHE_DIR / project / run_id
    rew_path = run_cache_dir / "my_ep_rewards_hist.npy"
    steps_path = run_cache_dir / "my_ep_steps_hist.npy"
    times_path = run_cache_dir / "my_ep_times_hist.npy"

    if not force_refresh and rew_path.is_file() and steps_path.is_file() and times_path.is_file():
        return np.load(rew_path), np.load(steps_path), np.load(times_path)

    ep_rew, steps, times = get_rew_steps_times(entity=ENTITY, project=project, run_id=run_id)
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(rew_path, ep_rew)
    np.save(steps_path, steps)
    np.save(times_path, times)
    return ep_rew, steps, times


def clip_to_env_cap(ep_rew, steps, times, env_name, *, x_axis):
    """Clip a run's raw history to the env's step/time safety ceiling.

    This never extends a curve -- it only ever shortens it. Individual runs
    in these sweeps stop at very different points (a wall-clock SLURM limit,
    not a shared step budget: e.g. Hopper runs log only ~1.5-3M of the
    10M-step ceiling); interpolate_group later pads each seed's curve with
    NaN past its own real extent rather than truncating every seed in a
    method down to the shortest one.
    """
    ep_rew = np.asarray(ep_rew, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if ep_rew.size == 0 or steps.size == 0:
        return None

    max_steps = None if x_axis == "time" else MAX_STEPS_PER_ENV.get(env_name)
    if max_steps is not None:
        keep = steps <= max_steps
        steps, ep_rew, times = steps[keep], ep_rew[keep], times[keep]
    if steps.size == 0:
        return None

    if x_axis == "time":
        rel_times = times - times[0]
        max_time_env = MAX_TIME_PER_ENV.get(env_name)
        if max_time_env is not None:
            keep = rel_times <= max_time_env
            rel_times, ep_rew, steps = rel_times[keep], ep_rew[keep], steps[keep]
        if rel_times.size == 0:
            return None
        return ep_rew, rel_times
    elif x_axis == "steps":
        return ep_rew, steps
    else:
        raise ValueError(f"Unknown x_axis: {x_axis}")


def interpolate_group(runs, *, n_points, smooth_window):
    """Build a shared grid spanning the longest contributing seed's real
    extent, then interpolate + smooth every seed onto its own portion of it.

    `runs` is a list of (ep_rew, x) raw-history pairs (x = steps or
    wall-time, already clipped to the env ceiling). A seed that stops
    logging before the grid's end is padded with NaN past its own last
    point rather than held flat or dropped -- so it never extrapolates, but
    it also never truncates the whole method down to its shortest seed.
    Downstream aggregation (compute_center_and_band_dropout) ignores those
    NaNs, so the seed count driving the mean/band shrinks along the tail as
    shorter seeds end, instead of capping every curve at the shortest one.
    """
    runs = [(ep_rew, x) for ep_rew, x in runs if x.size > 0]
    if not runs:
        return None
    shared_max_x = max(x[-1] for _, x in runs)
    if shared_max_x <= 0:
        return None
    grid = np.linspace(0, shared_max_x, num=n_points)
    curves = []
    for ep_rew, x in runs:
        n_valid = max(int(np.searchsorted(grid, x[-1], side="right")), 1)
        curve = moving_average(np.interp(grid[:n_valid], x, ep_rew), smooth_window)
        if n_valid < n_points:
            curve = np.concatenate([curve, np.full(n_points - n_valid, np.nan)])
        curves.append(curve)
    return curves, grid


def compute_center_and_band_dropout(values, *, center_stat="mean", band="std", n_resamples=10_000, seed=0):
    """Like plotter.compute_center_and_band, but tolerant of the NaN tail
    interpolate_group leaves once a seed stops logging: at each grid point
    the statistic is taken over whichever seeds are still finite there, so
    the effective seed count drops (n, n-1, ...) as shorter seeds end,
    instead of the whole curve being cut to the shortest seed.
    """
    values = np.asarray(values, dtype=np.float64)
    center_stat = normalize_center_stat(center_stat)
    band = normalize_band(band)
    statistic = get_center_statistic(center_stat)

    n_points = values.shape[1]
    center = np.full(n_points, np.nan)
    lower = np.full(n_points, np.nan)
    upper = np.full(n_points, np.nan)
    rng = np.random.default_rng(seed) if band == "95ci" else None

    for i in range(n_points):
        col = values[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        center[i] = statistic(col)
        if band == "std":
            spread = col.std()
            lower[i], upper[i] = center[i] - spread, center[i] + spread
        elif col.size == 1:
            lower[i] = upper[i] = center[i]
        else:
            estimates = statistic(col[rng.integers(col.size, size=(n_resamples, col.size))], axis=1)
            lower[i], upper[i] = np.percentile(estimates, [2.5, 97.5])

    return center, lower, upper


def build_data(
    envs=None,
    *,
    critic_reward_shapping=True,
    x_axis="steps",
    n_points=1000,
    smooth_window=25,
    force_refresh=False,
):
    envs = envs or list(PROJECTS)
    data = {}
    for env_name in envs:
        project = PROJECTS[env_name]
        print(f"Fetching run list for {env_name}: {ENTITY}/{project}")
        runs = list(API.runs(f"{ENTITY}/{project}", per_page=200))

        raw_by_method = {method: [] for method in METHOD_ORDER}

        for run in runs:
            if run.state != "finished" or run.id in EXCLUDED_RUN_IDS:
                continue
            cfg = dict(run.config)
            otil = cfg.get("agent", {}).get("otil", {})
            if otil.get("critic_reward_shapping") != critic_reward_shapping:
                continue

            critic_disabled = otil.get("critic_disabled")
            reward_mapping = otil.get("critic_reward_mapping")
            cost_type = otil.get("loss_ot_cost_type")
            if critic_disabled is None or reward_mapping is None or cost_type is None:
                continue

            method = method_name(critic_disabled, reward_mapping, cost_type)
            if method not in raw_by_method:
                warnings.warn(
                    f"{env_name}/{run.id}: unrecognized combo critic_disabled={critic_disabled}, "
                    f"critic_reward_mapping={reward_mapping}, loss_ot_cost_type={cost_type} -- skipping",
                    stacklevel=2,
                )
                continue

            ep_rew, steps, times = fetch_run_history(project, run.id, force_refresh=force_refresh)
            clipped = clip_to_env_cap(ep_rew, steps, times, env_name, x_axis=x_axis)
            if clipped is None:
                continue
            raw_by_method[method].append(clipped)

        env_data = {}
        for method in METHOD_ORDER:
            built = interpolate_group(raw_by_method[method], n_points=n_points, smooth_window=smooth_window)
            if built is None:
                env_data[method] = {"data": [], "x": None}
                continue
            curves, grid = built
            env_data[method] = {"data": curves, "x": grid}

        data[env_name] = env_data
        for method in METHOD_ORDER:
            method_data = env_data[method]
            n_seeds = len(method_data["data"])
            reached = f", reaches {method_data['x'][-1]:.3g}" if method_data["x"] is not None else ""
            print(f"  {method}: {n_seeds} seed curve(s){reached}")

    return data


def build_final_return_rows(data, *, center_stat="mean", band="std", normalize_expert=True):
    """Final return per env/method: center +/- spread (std band) or exact
    [lower, upper] (95ci band) across seeds, each seed contributing its own
    last logged (interpolated) value -- not a value at a shared grid point
    -- since seeds stop at different lengths.
    """
    rows = []
    for env_name, env_data in data.items():
        expert_mean = EXPERTS.get(env_name, {}).get("mean") if EXPERTS else None
        for method in METHOD_ORDER:
            method_data = env_data.get(method)
            if not method_data or not method_data["data"]:
                continue
            values = np.vstack(method_data["data"])
            if normalize_expert and expert_mean:
                values = values / expert_mean
            finite_mask = np.isfinite(values)
            last_idx = finite_mask.shape[1] - 1 - np.argmax(finite_mask[:, ::-1], axis=1)
            final_values = values[np.arange(values.shape[0]), last_idx][:, None]
            center, lower, upper = compute_center_and_band(final_values, center_stat=center_stat, band=band)
            rows.append(
                {
                    "environment": env_name,
                    "environment_name": ENV_NAMES.get(env_name, env_name),
                    "method": method,
                    "center": float(center[0]),
                    "lower": float(lower[0]),
                    "upper": float(upper[0]),
                    "n_runs": int(values.shape[0]),
                    "normalized_by_expert": bool(normalize_expert and expert_mean),
                }
            )
    return rows


def save_final_return_table(
    data, output_stem, *, center_stat="mean", band="std", critic_reward_shapping=True, normalize_expert=True
):
    rows = build_final_return_rows(data, center_stat=center_stat, band=band, normalize_expert=normalize_expert)
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

    present_envs = list(dict.fromkeys(row["environment_name"] for row in rows))
    preferred_envs = ["Hopper", "Ant", "Humanoid", "SNU Humanoid"]
    env_names = [env for env in preferred_envs if env in present_envs]
    env_names.extend(env for env in present_envs if env not in env_names)

    present_methods = {row["method"] for row in rows}
    method_names = [method for method in METHOD_ORDER if method in present_methods]

    values_by_env_method = {(row["environment_name"], row["method"]): row for row in rows}

    decimals = 3 if normalize_expert else 2

    def format_cell(row):
        if row is None:
            return "--"
        if normalize_band(band) == "std":
            spread = max(row["center"] - row["lower"], row["upper"] - row["center"])
            return rf"${row['center']:.{decimals}f} \pm {spread:.{decimals}f}$"
        return rf"${row['center']:.{decimals}f}\;[{row['lower']:.{decimals}f}, {row['upper']:.{decimals}f}]$"

    return_desc = "final returns normalized by each environment's expert return" if normalize_expert else "final returns"
    center_label = {"mean": "mean", "median": "median", "iqm": "interquartile mean"}[normalize_center_stat(center_stat)]
    stat_desc = (
        f"{center_label} " r"$\pm$ standard deviation across seeds"
        if normalize_band(band) == "std"
        else f"{center_label} with a 95\\% bootstrap confidence interval across seeds, shown as [lower, upper]"
    )
    caption = (
        f"Reward-shaping and critic ablation. Values are {return_desc}, reported as {stat_desc}. Rows use "
        r"\texttt{agent.otil.critic\_reward\_shapping=" + str(critic_reward_shapping) + "}."
    )

    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{0.35em}",
        r"\begin{tabular}{l" + "c" * len(env_names) + "}",
        r"  \toprule",
        "  Method & " + " & ".join(_latex_escape(name) for name in env_names) + r" \\",
        r"  \midrule",
    ]
    for index, method in enumerate(method_names):
        cells = [format_cell(values_by_env_method.get((env, method))) for env in env_names]
        latex_lines.append("  " + _latex_escape(method) + " & " + " & ".join(cells) + r" \\")
        if method.startswith("no-critic") and index != len(method_names) - 1:
            latex_lines.append(r"  \midrule")
    latex_lines.extend(
        [
            r"  \bottomrule",
            r"\end{tabular}",
            r"\vspace{4pt}",
            r"\caption{" + caption + "}",
            r"\label{tab:reward_ablation}",
            r"\end{table}",
        ]
    )
    tex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    return csv_path, tex_path


def _shared_legend(target, handles, labels, order=ALGOS, **legend_kwargs):
    if not handles:
        return
    map_lh = {label: handle for handle, label in zip(handles, labels)}
    ordered_labels = [algo for algo in order if algo in map_lh]
    ordered_handles = [map_lh[label] for label in ordered_labels]
    target.legend(ordered_handles, ordered_labels, frameon=False, **legend_kwargs)


def plot_training_curves(
    data,
    output_path,
    *,
    x_axis="steps",
    center_stat="mean",
    band="std",
    n_cols=2,
    add_expert_line=True,
    normalize_expert=True,
):
    env_order = [env for env in ("hopper", "ant", "humanoid", "snu_humanoid") if env in data]
    if not env_order:
        return

    n_envs = len(env_order)
    n_cols = min(n_cols, n_envs)
    n_rows = math.ceil(n_envs / n_cols)
    n_legend_cols = 6
    legend_rows = math.ceil(len(ALGOS) / n_legend_cols)
    fig = plt.figure(figsize=(6.5 * n_cols, 4.5 * n_rows + 0.35 * legend_rows), layout="constrained")
    grid = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[0.35 * legend_rows] + [4.5] * n_rows)
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.set_axis_off()
    axes = [fig.add_subplot(grid[1 + idx // n_cols, idx % n_cols]) for idx in range(n_envs)]

    legend_handles, legend_labels = None, None
    for ax_idx, env_name in enumerate(env_order):
        ax = axes[ax_idx]
        env_data = data[env_name]

        if not any(method_data["data"] for method_data in env_data.values()):
            ax.set_visible(False)
            continue

        x_label = "Relative Time (h)" if x_axis == "time" else "Steps"

        expert_mean = EXPERTS.get(env_name, {}).get("mean") if EXPERTS else None
        do_normalize = normalize_expert and expert_mean
        if add_expert_line and expert_mean is not None:
            expert_y = 1.0 if do_normalize else expert_mean
            ax.axhline(expert_y, color=COLOR["Expert"], linestyle=get_linestyle("Expert"), linewidth=1, label="Expert")

        for method in METHOD_ORDER:
            method_data = env_data.get(method)
            if not method_data or not method_data["data"]:
                continue
            x = method_data["x"]
            if x_axis == "time":
                x = convert_seconds_to_hours(x)
            values = np.vstack(method_data["data"])
            mean, lower, upper = compute_center_and_band_dropout(values, center_stat=center_stat, band=band)
            if do_normalize:
                mean, lower, upper = mean / expert_mean, lower / expert_mean, upper / expert_mean
            color = COLOR[method]
            ax.plot(x, mean, label=method, color=color, linestyle=get_linestyle(method), linewidth=get_linewidth(method))
            ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)

        ax.set_title(ENV_NAMES.get(env_name, env_name))
        ax.set_xlabel(x_label)
        ax.margins(x=0)
        ax.spines[["right", "top"]].set_visible(False)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    _shared_legend(legend_ax, legend_handles, legend_labels, loc="center", ncol=n_legend_cols, fontsize=8.5)
    fig.supylabel("Normalized Return" if normalize_expert else "Return", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_env_curves(
    env_name,
    env_data,
    output_path,
    *,
    x_axis="steps",
    center_stat="mean",
    band="std",
    add_expert_line=True,
    normalize_expert=True,
):
    if not any(method_data["data"] for method_data in env_data.values()):
        return

    x_label = "Relative Time (h)" if x_axis == "time" else "Steps"

    n_legend_cols = 2
    legend_rows = math.ceil(len(ALGOS) / n_legend_cols)
    fig = plt.figure(figsize=(6.5, 4.5 + 0.22 * legend_rows), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[0.22 * legend_rows, 4.5])
    legend_ax = fig.add_subplot(grid[0])
    legend_ax.set_axis_off()
    ax = fig.add_subplot(grid[1])

    expert_mean = EXPERTS.get(env_name, {}).get("mean") if EXPERTS else None
    do_normalize = normalize_expert and expert_mean
    if add_expert_line and expert_mean is not None:
        expert_y = 1.0 if do_normalize else expert_mean
        ax.axhline(expert_y, color=COLOR["Expert"], linestyle=get_linestyle("Expert"), linewidth=1, label="Expert")

    for method in METHOD_ORDER:
        method_data = env_data.get(method)
        if not method_data or not method_data["data"]:
            continue
        x = method_data["x"]
        if x_axis == "time":
            x = convert_seconds_to_hours(x)
        values = np.vstack(method_data["data"])
        mean, lower, upper = compute_center_and_band_dropout(values, center_stat=center_stat, band=band)
        if do_normalize:
            mean, lower, upper = mean / expert_mean, lower / expert_mean, upper / expert_mean
        color = COLOR[method]
        ax.plot(x, mean, label=method, color=color, linestyle=get_linestyle(method), linewidth=get_linewidth(method))
        ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)

    ax.set_title(ENV_NAMES.get(env_name, env_name))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized Return" if normalize_expert else "Return")
    ax.margins(x=0)
    ax.spines[["right", "top"]].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    # Two columns, critics left / no-critics right so each pair shares a row;
    # Expert fills the spare slot at the bottom of the left column.
    single_env_order = CRITIC_METHODS + ["Expert"] + NO_CRITIC_METHODS
    _shared_legend(
        legend_ax, handles, labels, order=single_env_order, loc="center", ncol=n_legend_cols, fontsize=8.5
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(
    envs=None,
    critic_reward_shapping=True,
    x_axis="steps",
    center_stat="mean",
    band="std",
    n_points=1000,
    smooth_window=25,
    force_refresh=False,
    normalize_expert=True,
):
    data = build_data(
        envs=envs,
        critic_reward_shapping=critic_reward_shapping,
        x_axis=x_axis,
        n_points=n_points,
        smooth_window=smooth_window,
        force_refresh=force_refresh,
    )

    suffix = "_normalized" if normalize_expert else ""

    table_stem = FOLDER_TO_SAVE_PLOTS / f"critic_no_critic_table_{center_stat}{band}_{x_axis}{suffix}"
    save_final_return_table(
        data,
        table_stem,
        center_stat=center_stat,
        band=band,
        critic_reward_shapping=critic_reward_shapping,
        normalize_expert=normalize_expert,
    )

    plot_training_curves(
        data,
        FOLDER_TO_SAVE_PLOTS / f"critic_no_critic_curves_{center_stat}{band}_{x_axis}{suffix}.png",
        x_axis=x_axis,
        center_stat=center_stat,
        band=band,
        normalize_expert=normalize_expert,
    )

    per_env_dir = FOLDER_TO_SAVE_PLOTS / "per_env"
    for env_name, env_data in data.items():
        plot_single_env_curves(
            env_name,
            env_data,
            per_env_dir / f"{env_name}_{center_stat}{band}_{x_axis}{suffix}.png",
            x_axis=x_axis,
            center_stat=center_stat,
            band=band,
            normalize_expert=normalize_expert,
        )

    return data


if __name__ == "__main__":
    # Mirrors plotter.py's own __main__ loop: regenerate every stat/band
    # combo (and both normalized/raw) on every run, instead of only the
    # meanstd default -- so the report never silently goes stale relative
    # to the others.
    for center_stat, band in [("mean", "std"), ("mean", "95ci"), ("median", "95ci")]:
        for normalize_expert in (True, False):
            main(
                critic_reward_shapping=True,
                x_axis="steps",
                center_stat=center_stat,
                band=band,
                force_refresh=False,
                normalize_expert=normalize_expert,
            )
