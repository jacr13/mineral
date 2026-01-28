from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from utils import load_yaml, save_yaml

ENVS = ["hopper", "ant", "humanoid", "snu_humanoid"]
ALGOS = ["ILD", "OTIL"]
VARIANTS = ["SAPO", "SHAC"]
PROJECT_TEMPLATES = {
    "ILD": "{algo}_{variant}-sweep-dflex_{env_tag}-slurm",
    "OTIL": "{algo}_{variant}-dflex_{env_tag}-slurm-new",
}

ENV_NAMES = {
    "hopper": "Hopper",
    "ant": "Ant",
    "humanoid": "Humanoid",
    "snu_humanoid": "SNU Humanoid",
}

EXPERTS = {
    "hopper": {"mean": 4812.96875, "std": 7.324751853942871},
    "ant": {"mean": 9329.505859375, "std": 37.58782196044922},
    "humanoid": {"mean": 8225.2177734375, "std": 81.0524673461914},
    "snu_humanoid": {"mean": 6498.921875, "std": 49.13310623168945},
}

ENTITY = "jacr"

LOAD_FROM_WANDB = False
HORIZON_LEN_FILTER = 32  # e.g. 32 or [32]
PLOT_KINDS = ["violin", "box"]  # options: "violin", "box"
PLOT_TRENDS = True
PLOT_DELTA = True
PLOT_SAPO_ILD_OTIL_TREND = True
PLOT_OTIL_SHAC_VS_SAPO = True
PRINT_OTIL_LATEX_TABLE = True
PRINT_FOCUS_LATEX_TABLE = True
SAPO_ILD_HORIZON_FILTER = 32
SAPO_OTIL_HORIZON_FILTER = None
SAPO_VARIANCE = "std"  # options: "std", "sem"
SAPO_NORMALIZE_RETURN = True
OTIL_VARIANT_NORMALIZE_RETURN = True
OTIL_LATEX_NORMALIZE_RETURN = True
FOCUS_LATEX_NORMALIZE_RETURN = True
FILTER_OTIL_BY_LOSS = True
OTIL_LOSS_OT_COST_TYPE = "l2"
OTIL_IMITATION_LOSS_TYPE = "l2"
OUTPUT_EXTS = ["png"]

sns.set_theme(style="whitegrid")
sns.set(rc={"axes.facecolor": "#f5f5f5"})


def load_or_refresh_data(path2data):
    data = {}
    if path2data.exists():
        data = load_yaml(path2data)

    if not LOAD_FROM_WANDB:
        if not data:
            raise FileNotFoundError(f"Missing data file: {path2data}")
        return data

    import wandb
    from wandb_api import get_rew_steps_times  # lazy import (wandb dependency)

    for env in ENVS:
        if env in data:
            continue
        data[env] = {}
        for algo in ALGOS:
            data[env][algo] = {}
            for variant in VARIANTS:
                data[env][algo][variant] = {}
                project = PROJECT_TEMPLATES[algo].format(algo=algo, variant=variant, env_tag=env)
                print(project)

                api = wandb.Api()
                runs = api.runs(f"{ENTITY}/{project}")

                for run in runs:
                    print(run.group)

                    if run.group not in data[env][algo][variant]:
                        data[env][algo][variant][run.group] = []

                    ep_rew, _, _ = get_rew_steps_times(
                        entity=ENTITY,
                        project=project,
                        run_id=run.id,
                    )

                    try:
                        imitation_loss_type = run.config["agent"]["otil"]["imitation_loss_type"]
                        loss_ot_cost_type = run.config["agent"]["otil"]["loss_ot_cost_type"]
                    except KeyError:
                        imitation_loss_type = None
                        loss_ot_cost_type = None

                    data[env][algo][variant][run.group].append(
                        {
                            "run_id": run.id,
                            "name": run.name,
                            "project": run.project,
                            "entity": run.entity,
                            "imitation_loss_type": imitation_loss_type,
                            "loss_ot_cost_type": loss_ot_cost_type,
                            "horizon_len": run.config["agent"]["shac"]["horizon_len"],
                            "demos": run.config["agent"][algo.lower()]["demos"]["n_trajs"],
                            "logdir": run.config["logdir"],
                            "ep_rew": float(ep_rew.max()),
                        }
                    )

        save_yaml(path2data, data)

    return data


def build_dataframe(data):
    rows = []
    for env, env_data in data.items():
        for algo, algo_data in env_data.items():
            for variant, variant_data in algo_data.items():
                for group, runs in variant_data.items():
                    if not runs:
                        continue
                    for run in runs:
                        if run is None:
                            continue
                        demos = run.get("demos")
                        ep_rew = run.get("ep_rew")
                        if demos is None or ep_rew is None:
                            continue
                        rows.append(
                            {
                                "env": env,
                                "env_name": ENV_NAMES.get(env, env),
                                "algo": algo,
                                "variant": variant,
                                "group": group,
                                "run_id": run.get("run_id"),
                                "imitation_loss_type": run.get("imitation_loss_type"),
                                "loss_ot_cost_type": run.get("loss_ot_cost_type"),
                                "horizon_len": run.get("horizon_len"),
                                "demos": int(demos),
                                "ep_rew": float(ep_rew),
                            }
                        )
    return pd.DataFrame(rows)


def apply_filters(df):
    if ALGOS:
        df = df[df["algo"].isin(ALGOS)]
    if VARIANTS:
        df = df[df["variant"].isin(VARIANTS)]
    if HORIZON_LEN_FILTER is not None:
        allowed = HORIZON_LEN_FILTER
        if not isinstance(allowed, (list, tuple, set, np.ndarray)):
            allowed = [allowed]
        df = df[df["horizon_len"].isin(allowed)]
    if FILTER_OTIL_BY_LOSS:
        non_otil = df[df["algo"] != "OTIL"]
        otil = df[df["algo"] == "OTIL"].copy()
        if not otil.empty:
            otil["imitation_loss_type"] = otil["imitation_loss_type"].astype("string").str.lower()
            otil["loss_ot_cost_type"] = otil["loss_ot_cost_type"].astype("string").str.lower()
            otil = otil[
                (otil["imitation_loss_type"] == OTIL_IMITATION_LOSS_TYPE.lower())
                & (otil["loss_ot_cost_type"] == OTIL_LOSS_OT_COST_TYPE.lower())
            ]
        if non_otil.empty:
            df = otil.copy()
        elif otil.empty:
            df = non_otil.copy()
        else:
            df = pd.concat([non_otil, otil], ignore_index=True)
    return df


def get_env_order(df):
    order = []
    present = set(df["env_name"].unique())
    for env in ENVS:
        name = ENV_NAMES.get(env, env)
        if name in present:
            order.append(name)
    for name in sorted(present - set(order)):
        order.append(name)
    return order


def get_algo_order(df):
    order = [algo for algo in ALGOS if algo in set(df["algo"].unique())]
    for algo in sorted(set(df["algo"].unique()) - set(order)):
        order.append(algo)
    return order


def get_expert_mean(env_name):
    for env_key, pretty in ENV_NAMES.items():
        if pretty == env_name:
            return EXPERTS.get(env_key, {}).get("mean")
    return EXPERTS.get(env_name, {}).get("mean")


def save_figure(fig, output_dir, name_base):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in OUTPUT_EXTS:
        fig.savefig(output_dir / f"{name_base}.{ext}", dpi=200, bbox_inches="tight")


def plot_distributions(df, output_dir):
    demo_order = sorted(df["demos"].unique())
    df = df.copy()
    df["demos"] = pd.Categorical(df["demos"], categories=demo_order, ordered=True)
    env_order = get_env_order(df)
    algo_order = get_algo_order(df)

    for kind in PLOT_KINDS:
        grid = sns.catplot(
            data=df,
            kind=kind,
            x="demos",
            y="ep_rew",
            hue="variant",
            col="env_name",
            row="algo",
            height=3.3,
            aspect=1.2,
            sharey=False,
            hue_order=VARIANTS,
            col_order=env_order,
            row_order=algo_order,
        )
        grid.set_axis_labels("# Trajectories", "Episode Reward")
        grid.set_titles("{row_name} | {col_name}")
        if grid._legend is not None:
            grid._legend.set_title("Variant")
        for ax in grid.axes.flat:
            ax.grid(True, axis="y", alpha=0.25)
        save_figure(grid.fig, output_dir, f"shac_vs_sapo_{kind}_by_trajs")
        plt.close(grid.fig)


def plot_trends(df, output_dir):
    demo_order = sorted(df["demos"].unique())
    env_order = get_env_order(df)
    algo_order = get_algo_order(df)
    summary = (
        df.groupby(["env_name", "algo", "variant", "demos"], observed=True)["ep_rew"]
        .median()
        .reset_index(name="median_ep_rew")
    )
    summary["demos"] = pd.Categorical(summary["demos"], categories=demo_order, ordered=True)

    grid = sns.relplot(
        data=summary,
        kind="line",
        x="demos",
        y="median_ep_rew",
        hue="variant",
        col="env_name",
        row="algo",
        marker="o",
        height=3.3,
        aspect=1.2,
        facet_kws={"sharey": False},
        hue_order=VARIANTS,
        col_order=env_order,
        row_order=algo_order,
    )
    grid.set_axis_labels("# Trajectories", "Median Episode Reward")
    grid.set_titles("{row_name} | {col_name}")
    if grid._legend is not None:
        grid._legend.set_title("Variant")
    for ax in grid.axes.flat:
        ax.grid(True, axis="y", alpha=0.25)
    save_figure(grid.fig, output_dir, "shac_vs_sapo_trend_median")
    plt.close(grid.fig)


def plot_delta(df, output_dir):
    env_order = get_env_order(df)
    algo_order = get_algo_order(df)
    summary = (
        df.groupby(["env_name", "algo", "variant", "demos"], observed=True)["ep_rew"]
        .median()
        .reset_index(name="median_ep_rew")
    )
    pivot = summary.pivot_table(
        index=["env_name", "algo", "demos"],
        columns="variant",
        values="median_ep_rew",
    ).reset_index()
    if "SHAC" not in pivot or "SAPO" not in pivot:
        return
    pivot["delta"] = pivot["SHAC"] - pivot["SAPO"]
    pivot = pivot.dropna(subset=["delta"])
    if pivot.empty:
        return
    pivot["better"] = np.where(pivot["delta"] >= 0, "SHAC better", "SAPO better")

    demo_order = sorted(pivot["demos"].unique())
    pivot["demos"] = pd.Categorical(pivot["demos"], categories=demo_order, ordered=True)

    palette = {"SHAC better": "#2ca02c", "SAPO better": "#d62728"}
    grid = sns.catplot(
        data=pivot,
        kind="bar",
        x="demos",
        y="delta",
        hue="better",
        col="env_name",
        row="algo",
        height=3.3,
        aspect=1.2,
        sharey=False,
        palette=palette,
        dodge=False,
        col_order=env_order,
        row_order=algo_order,
    )
    grid.set_axis_labels("# Trajectories", "Median Reward Δ (SHAC - SAPO)")
    grid.set_titles("{row_name} | {col_name}")
    if grid._legend is not None:
        grid._legend.set_title("Winner")
    for ax in grid.axes.flat:
        ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
        ax.grid(True, axis="y", alpha=0.25)
    save_figure(grid.fig, output_dir, "shac_vs_sapo_delta_median")
    plt.close(grid.fig)


def plot_sapo_ild_otil_trend(df, output_dir):
    df = df[(df["variant"] == "SAPO") & (df["algo"].isin(["ILD", "OTIL"]))]
    if df.empty:
        return

    ild_mask = df["algo"] == "ILD"
    if SAPO_ILD_HORIZON_FILTER is not None:
        df = pd.concat(
            [
                df[ild_mask & df["horizon_len"].isin(np.atleast_1d(SAPO_ILD_HORIZON_FILTER))],
                df[~ild_mask],
            ],
            ignore_index=True,
        )

    if SAPO_OTIL_HORIZON_FILTER is not None:
        otil_mask = df["algo"] == "OTIL"
        df = pd.concat(
            [
                df[~otil_mask],
                df[otil_mask & df["horizon_len"].isin(np.atleast_1d(SAPO_OTIL_HORIZON_FILTER))],
            ],
            ignore_index=True,
        )

    if df.empty:
        return

    env_order = get_env_order(df)
    palette = {"ILD": "#1f77b4", "OTIL": "#ff7f0e"}
    marker_map = {"ILD": "o", "OTIL": "s"}
    label_map = {
        "ILD": f"ILD (h={SAPO_ILD_HORIZON_FILTER})" if SAPO_ILD_HORIZON_FILTER is not None else "ILD",
        "OTIL": "OTIL",
    }

    for env_name in env_order:
        env_df = df[df["env_name"] == env_name].copy()
        if env_df.empty:
            continue
        expert_mean = get_expert_mean(env_name)
        if SAPO_NORMALIZE_RETURN and expert_mean:
            env_df["norm_ep_rew"] = env_df["ep_rew"] / expert_mean
        else:
            env_df["norm_ep_rew"] = env_df["ep_rew"]

        demo_order = sorted(env_df["demos"].unique())
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=env_df,
            x="demos",
            y="norm_ep_rew",
            hue="algo",
            order=demo_order,
            hue_order=["ILD", "OTIL"],
            palette=palette,
            ax=ax,
        )
        sns.pointplot(
            data=env_df,
            x="demos",
            y="norm_ep_rew",
            hue="algo",
            order=demo_order,
            hue_order=["ILD", "OTIL"],
            markers=[marker_map["ILD"], marker_map["OTIL"]],
            linestyles=["-", "-"],
            dodge=0.4,
            errorbar=None,
            palette=palette,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()

        handles = [
            Line2D([0], [0], color=palette["ILD"], marker=marker_map["ILD"], linestyle="-", label=label_map["ILD"]),
            Line2D([0], [0], color=palette["OTIL"], marker=marker_map["OTIL"], linestyle="-", label=label_map["OTIL"]),
        ]
        ax.legend(
            handles=handles,
            title="Algo",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=2,
        )
        ax.set_title(f"{env_name} | SAPO")
        ax.set_xlabel("# Trajectories")
        ax.set_ylabel("Expert Normalized Return")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        env_slug = env_name.lower().replace(" ", "_")
        save_figure(fig, output_dir, f"sapo_ild_otil_box_{env_slug}")
        plt.close(fig)


def plot_otil_shac_vs_sapo(df, output_dir):
    df = df[(df["algo"] == "OTIL") & (df["variant"].isin(["SAPO", "SHAC"]))]
    if df.empty:
        return

    env_order = get_env_order(df)
    palette = {"SAPO": "#1f77b4", "SHAC": "#ff7f0e"}
    marker_map = {"SAPO": "o", "SHAC": "s"}

    for env_name in env_order:
        env_df = df[df["env_name"] == env_name].copy()
        if env_df.empty:
            continue
        expert_mean = get_expert_mean(env_name)
        if OTIL_VARIANT_NORMALIZE_RETURN and expert_mean:
            env_df["norm_ep_rew"] = env_df["ep_rew"] / expert_mean
        else:
            env_df["norm_ep_rew"] = env_df["ep_rew"]

        print(env_df)

        demo_order = sorted(env_df["demos"].unique())
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=env_df,
            x="demos",
            y="norm_ep_rew",
            hue="variant",
            order=demo_order,
            hue_order=["SAPO", "SHAC"],
            palette=palette,
            ax=ax,
        )
        sns.pointplot(
            data=env_df,
            x="demos",
            y="norm_ep_rew",
            hue="variant",
            order=demo_order,
            hue_order=["SAPO", "SHAC"],
            markers=[marker_map["SAPO"], marker_map["SHAC"]],
            linestyles=["-", "-"],
            dodge=0.4,
            errorbar=None,
            palette=palette,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()

        label_tmp = f"FOCUS-{OTIL_IMITATION_LOSS_TYPE.upper()}-{OTIL_LOSS_OT_COST_TYPE.replace('cosine', 'cos').upper()}"
        label_tmp = label_tmp.replace("-L2-L2", "-L2")

        handles = [
            Line2D(
                [0],
                [0],
                color=palette["SAPO"],
                marker=marker_map["SAPO"],
                linestyle="-",
                label=f"{label_tmp} w SAPO",
            ),
            Line2D(
                [0],
                [0],
                color=palette["SHAC"],
                marker=marker_map["SHAC"],
                linestyle="-",
                label=f"{label_tmp} w SHAC",
            ),
        ]
        ax.legend(
            handles=handles,
            title="",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=2,
        )
        # ax.set_title(f"{env_name} | OTIL")
        ax.set_xlabel("# Trajectories")
        ax.set_ylabel("Expert Normalized Return" if OTIL_VARIANT_NORMALIZE_RETURN else "Episode Reward")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        env_slug = env_name.lower().replace(" ", "_")
        save_figure(fig, output_dir, f"focus-{OTIL_IMITATION_LOSS_TYPE}-{OTIL_LOSS_OT_COST_TYPE}_shac_vs_sapo_box_{env_slug}")
        plt.close(fig)


def format_mean_std(mean_val, std_val):
    if pd.isna(mean_val):
        return "-"
    if pd.isna(std_val):
        return f"{mean_val:.3f}"
    return f"{mean_val:.3f} \\pm {std_val:.3f}"


def print_otil_shac_vs_sapo_table(df, output_dir):
    df = df[(df["algo"] == "OTIL") & (df["variant"].isin(["SAPO", "SHAC"]))]
    if df.empty:
        return

    df = df.copy()
    if OTIL_LATEX_NORMALIZE_RETURN:
        df["norm_ep_rew"] = df.apply(
            lambda row: row["ep_rew"] / get_expert_mean(row["env_name"]) if get_expert_mean(row["env_name"]) else np.nan,
            axis=1,
        )
        value_col = "norm_ep_rew"
    else:
        value_col = "ep_rew"

    stats = df.groupby(["env_name", "demos", "variant"], observed=True)[value_col].agg(["mean", "std", "count"]).reset_index()
    if stats.empty:
        return

    mean_pivot = stats.pivot_table(index=["env_name", "demos"], columns="variant", values="mean")
    std_pivot = stats.pivot_table(index=["env_name", "demos"], columns="variant", values="std")

    rows = []
    for env_name, demos in mean_pivot.index:
        mean_sapo = mean_pivot.loc[(env_name, demos)].get("SAPO")
        mean_shac = mean_pivot.loc[(env_name, demos)].get("SHAC")
        std_sapo = std_pivot.loc[(env_name, demos)].get("SAPO")
        std_shac = std_pivot.loc[(env_name, demos)].get("SHAC")
        delta = mean_shac - mean_sapo if pd.notna(mean_shac) and pd.notna(mean_sapo) else np.nan
        rows.append(
            {
                "Environment": env_name,
                "# Traj": demos,
                "SAPO": format_mean_std(mean_sapo, std_sapo),
                "SHAC": format_mean_std(mean_shac, std_shac),
                "\\Delta (SHAC-SAPO)": "-" if pd.isna(delta) else f"{delta:.3f}",
            }
        )

    table = pd.DataFrame(rows)
    env_order = get_env_order(df)
    table["Environment"] = pd.Categorical(table["Environment"], categories=env_order, ordered=True)
    table = table.sort_values(["Environment", "# Traj"]).reset_index(drop=True)

    latex = table.to_latex(index=False, escape=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "otil_shac_vs_sapo_table.tex"
    output_path.write_text(latex, encoding="utf-8")
    print(latex)


def print_focus_shac_vs_sapo_table(df, output_dir):
    df = df[(df["algo"].isin(["FOCUS", "OTIL"])) & (df["variant"].isin(["SAPO", "SHAC"]))]
    if df.empty:
        return
    if (df["algo"] == "FOCUS").any():
        df = df[df["algo"] == "FOCUS"]
    else:
        df = df[df["algo"] == "OTIL"]

    df = df.copy()
    if FOCUS_LATEX_NORMALIZE_RETURN:
        df["norm_ep_rew"] = df.apply(
            lambda row: row["ep_rew"] / get_expert_mean(row["env_name"]) if get_expert_mean(row["env_name"]) else np.nan,
            axis=1,
        )
        value_col = "norm_ep_rew"
    else:
        value_col = "ep_rew"

    stats = df.groupby(["env_name", "demos", "variant"], observed=True)[value_col].agg(["mean", "std", "count"]).reset_index()
    if stats.empty:
        return

    stats["variant_label"] = stats["variant"].map({"SAPO": "FOCUS-sapo", "SHAC": "FOCUS-shac"})
    stats["value"] = stats.apply(lambda row: format_mean_std(row["mean"], row["std"]), axis=1)

    table = stats.pivot_table(
        index="env_name",
        columns=["variant_label", "demos"],
        values="value",
        aggfunc="first",
    )

    env_order = get_env_order(df)
    table = table.reindex(env_order)
    demo_order = sorted(df["demos"].unique())
    variant_order = ["FOCUS-sapo", "FOCUS-shac"]
    desired_cols = [(variant, demo) for variant in variant_order for demo in demo_order]
    table = table.reindex(columns=pd.MultiIndex.from_tuples(desired_cols))

    table.index.name = "Environment"
    latex = table.to_latex(escape=False, multicolumn=True, multirow=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "focus_shac_vs_sapo_table.tex"
    output_path.write_text(latex, encoding="utf-8")
    print(latex)


def main():
    path_dir = Path(__file__).resolve().parent / "shac_vs_sapo"
    path2data = path_dir / "data.yaml"
    output_dir = path_dir / "figures"

    data = load_or_refresh_data(path2data)
    df = build_dataframe(data)
    if df.empty:
        print("No data found in data.yaml.")
        return

    df = apply_filters(df)
    print(df)
    df.to_csv(path_dir / "filtered_data.csv", index=False)
    if df.empty:
        print("No data after filtering.")
        return

    if PLOT_KINDS:
        plot_distributions(df, output_dir)
    if PLOT_TRENDS:
        plot_trends(df, output_dir)
    if PLOT_DELTA:
        plot_delta(df, output_dir)
    if PLOT_SAPO_ILD_OTIL_TREND:
        plot_sapo_ild_otil_trend(df, output_dir)
    if PLOT_OTIL_SHAC_VS_SAPO:
        plot_otil_shac_vs_sapo(df, output_dir)
    if PRINT_OTIL_LATEX_TABLE:
        print_otil_shac_vs_sapo_table(df, output_dir)
    if PRINT_FOCUS_LATEX_TABLE:
        print_focus_shac_vs_sapo_table(df, output_dir)


if __name__ == "__main__":
    main()
