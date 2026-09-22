"""Best-of-K pool size ablation (OTIL/FOCUS-SAPO, OT-L2 vs OT-Cos): downloads
final eval returns from wandb, then produces a K-vs-return plot (raw and
expert-normalized) and a LaTeX/CSV final-return table with std or 95% CI
bands -- same shape as critic_no_critic.py, but the swept axis is
agent.otil.loss_best_of_k_k instead of critic_disabled/reward_mapping.

Sweeps live in one wandb project per environment (see PROJECTS below). Every
run carries these config fields under agent.otil:
    loss_best_of_k_k     int  -- 1, 2, 4, 8, 16
    loss_ot_cost_type    str  -- "l2" | "cosine"
Each (loss_best_of_k_k, loss_ot_cost_type) pair maps to one "method", e.g.
"k8-l2" or "k16-cos" (3 seeds each).
"""

import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from wandb_api import API

from plotter import (
    ENV_NAMES,
    EXPERTS,
    _latex_escape,
    compute_center_and_band,
    normalize_band,
    normalize_center_stat,
)

PLOTTER_DIR = Path(__file__).resolve().parent
FOLDER_TO_SAVE_PLOTS = PLOTTER_DIR / "bestofk_ablation"

ENTITY = "jacr"
ENVS = ["hopper", "ant", "humanoid", "snu_humanoid"]
PROJECTS = {env: f"bestofk_OTIL_SAPO-dflex_{env}-slurm-new" for env in ENVS}

K_VALUES = [1, 2, 4, 8, 16]
COST_LABEL = {"l2": "l2", "cosine": "cos"}

# Paired (k-l2, k-cos) per K, ascending K, so the table/legend put each pair
# next to each other for a direct comparison -- same idea as critic_no_critic's
# METHOD_ORDER pairing.
METHOD_ORDER = [f"k{k}-{COST_LABEL[cost]}" for k in K_VALUES for cost in ("l2", "cosine")]
ALGOS = ["Expert"] + METHOD_ORDER
ALGO_INDEX = {algo: idx for idx, algo in enumerate(ALGOS)}

# Same hex colors plotter.py already uses for these two OT variants, so this
# stays visually consistent with the project's other figures. K is encoded by
# marker/line (each K gets its own point on the x axis), not by a 3rd color.
COLOR = {"Expert": "#616161", "l2": "#479A5F", "cosine": "#A0C75C"}
LABEL = {"l2": "FOCUS-OT-L2", "cosine": "FOCUS-OT-Cos"}


def method_name(k, cost_type):
    return f"k{k}-{COST_LABEL.get(cost_type, cost_type)}"


def fetch_data(envs=None, force_refresh=False):
    """final_return[env][method] = list of per-seed final eval returns."""
    envs = envs or ENVS
    data = {}
    for env_name in envs:
        project = PROJECTS[env_name]
        print(f"Fetching run list for {env_name}: {ENTITY}/{project}")
        runs = list(API.runs(f"{ENTITY}/{project}", per_page=200))

        raw_by_method = {method: [] for method in METHOD_ORDER}
        skipped = 0
        for run in runs:
            if run.state != "finished":
                continue
            cfg = dict(run.config)
            otil = cfg.get("agent", {}).get("otil", {})
            k = otil.get("loss_best_of_k_k")
            cost_type = otil.get("loss_ot_cost_type")
            if k is None or cost_type is None:
                continue

            method = method_name(int(k), cost_type)
            if method not in raw_by_method:
                warnings.warn(
                    f"{env_name}/{run.id}: unrecognized combo loss_best_of_k_k={k}, "
                    f"loss_ot_cost_type={cost_type} -- skipping",
                    stacklevel=2,
                )
                continue

            final_return = run.summary.get("eval_scores/episode_rewards")
            if final_return is None:
                skipped += 1
                continue
            raw_by_method[method].append(float(final_return))

        data[env_name] = raw_by_method
        if skipped:
            print(f"  ({skipped} finished run(s) missing eval_scores/episode_rewards, skipped -- "
                  "see scripts/rerun_missing_bestofk_seeds.sh)")
        for method in METHOD_ORDER:
            print(f"  {method}: {len(raw_by_method[method])} seed(s)")

    return data


def build_final_return_rows(data, *, center_stat="mean", band="std", normalize_expert=True):
    rows = []
    for env_name, env_data in data.items():
        expert_mean = EXPERTS.get(env_name, {}).get("mean")
        for method in METHOD_ORDER:
            values = env_data.get(method, [])
            if not values:
                continue
            values = np.asarray(values, dtype=np.float64)
            if normalize_expert and expert_mean:
                values = values / expert_mean
            center, lower, upper = compute_center_and_band(values[:, None], center_stat=center_stat, band=band)
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


def save_final_return_table(data, output_stem, *, center_stat="mean", band="std", normalize_expert=True):
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

    return_desc = "final eval returns normalized by each environment's expert return" if normalize_expert else "final eval returns"
    center_label = {"mean": "mean", "median": "median", "iqm": "interquartile mean"}[normalize_center_stat(center_stat)]
    stat_desc = (
        f"{center_label} " r"$\pm$ standard deviation across seeds"
        if normalize_band(band) == "std"
        else f"{center_label} with a 95\\% bootstrap confidence interval across seeds, shown as [lower, upper]"
    )
    caption = (
        f"Best-of-K pool size ablation. Values are {return_desc}, reported as {stat_desc}. "
        r"K sweeps \texttt{agent.otil.loss\_best\_of\_k\_k} over " + str(K_VALUES) + "."
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
        if method.endswith("-cos") and index != len(method_names) - 1:
            latex_lines.append(r"  \midrule")
    latex_lines.extend(
        [
            r"  \bottomrule",
            r"\end{tabular}",
            r"\vspace{4pt}",
            r"\caption{" + caption + "}",
            r"\label{tab:bestofk_ablation}",
            r"\end{table}",
        ]
    )
    tex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    return csv_path, tex_path


def plot_k_ablation(data, output_path, *, center_stat="mean", band="std", normalize_expert=True):
    env_order = [env for env in ENVS if env in data]
    if not env_order:
        return

    fig, axes = plt.subplots(1, len(env_order), figsize=(5 * len(env_order), 4), sharex=True)
    if len(env_order) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, env_order):
        env_data = data[env_name]
        expert_mean = EXPERTS.get(env_name, {}).get("mean")
        do_normalize = normalize_expert and expert_mean

        for cost in ("l2", "cosine"):
            ks, centers, lowers, uppers = [], [], [], []
            for k in K_VALUES:
                values = env_data.get(method_name(k, cost), [])
                if not values:
                    continue
                values = np.asarray(values, dtype=np.float64)
                if do_normalize:
                    values = values / expert_mean
                center, lower, upper = compute_center_and_band(values[:, None], center_stat=center_stat, band=band)
                ks.append(k)
                centers.append(center[0])
                lowers.append(lower[0])
                uppers.append(upper[0])
            if not ks:
                continue
            centers, lowers, uppers = np.array(centers), np.array(lowers), np.array(uppers)
            yerr = np.vstack([centers - lowers, uppers - centers])
            ax.errorbar(
                ks, centers, yerr=yerr, marker="o", markersize=6, linewidth=2, capsize=3,
                label=LABEL[cost], color=COLOR[cost],
            )

        if do_normalize:
            ax.axhline(1.0, color=COLOR["Expert"], linestyle=":", linewidth=1, label="Expert")

        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Best-of-K pool size (K)")
        ax.set_title(ENV_NAMES.get(env_name, env_name))
        ax.spines[["right", "top"]].set_visible(False)

    center_label = {"mean": "Mean", "median": "Median", "iqm": "IQM"}[normalize_center_stat(center_stat)]
    band_desc = "+/- 1 SD" if normalize_band(band) == "std" else "95% bootstrap CI"
    y_label = f"{center_label} final return" + (" (normalized by expert)" if normalize_expert else "")
    axes[0].set_ylabel(y_label)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(f"OTIL/FOCUS-SAPO: effect of best-of-K pool size on final return ({band_desc} across 3 seeds)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main(envs=None, center_stat="mean", band="std", normalize_expert=True, force_refresh=False, data=None):
    data = data if data is not None else fetch_data(envs=envs, force_refresh=force_refresh)

    suffix = "_normalized" if normalize_expert else ""
    table_stem = FOLDER_TO_SAVE_PLOTS / f"bestofk_table_{center_stat}{band}{suffix}"
    save_final_return_table(data, table_stem, center_stat=center_stat, band=band, normalize_expert=normalize_expert)

    plot_k_ablation(
        data,
        FOLDER_TO_SAVE_PLOTS / f"bestofk_plot_{center_stat}{band}{suffix}.png",
        center_stat=center_stat,
        band=band,
        normalize_expert=normalize_expert,
    )
    return data


if __name__ == "__main__":
    # Mirrors critic_no_critic.py's __main__ loop: regenerate every
    # stat/band combo (and both normalized/raw) so the report never goes
    # stale relative to the others. Data is fetched once and reused.
    shared_data = fetch_data()
    for center_stat, band in [("mean", "std"), ("mean", "95ci"), ("median", "95ci")]:
        for normalize_expert in (True, False):
            main(center_stat=center_stat, band=band, normalize_expert=normalize_expert, data=shared_data)
