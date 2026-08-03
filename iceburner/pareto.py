"""Hypervolume tracking, Pareto-front extraction, and Pareto plots.

Requires ``torch`` and ``botorch`` (used for hypervolume and
non-dominated-set computation), in addition to the usual scientific
Python stack.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

from . import scaling

TKWARGS = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Campaign-wide objective/input naming, shared across every notebook.
DEFAULT_OBJECTIVE_COLS = [
    "Y_TDT",
    "Y_rhoRDT",
    "Y_minus_peak_IFAR",
    "Y_minus_current",
]

DEFAULT_OBJECTIVE_LABELS = {
    "Y_TDT": "TDT",
    "Y_rhoRDT": r"$\rho R_{DT}$",
    "Y_minus_peak_IFAR": "- peak IFAR",
    "Y_minus_current": "- current",
}

DEFAULT_INPUT_COLS = ["f_laser", "f_R_outer", "f_Pi", "f_Be", "f_rho", "current"]

DEFAULT_INPUT_LABELS = {
    "f_laser": "f_laser",
    "f_R_outer": "f_R_outer",
    "f_Pi": "f_Pi",
    "f_Be": "f_Be",
    "f_rho": "f_rho",
    "current": "current",
}

DEFAULT_OTHER_COLS = [
    "P_YDT",
    "P_TDT",
    "P_BT",
    "P_imp_vel",
    "P_rhoRDT",
    "P_rhoRBe",
    "P_peak_IFAR",
]

DEFAULT_COLUMN_MAP = {
    "temperature": "Y_TDT",
    "rhoR": "Y_rhoRDT",
    "minus_peak_IFAR": "Y_minus_peak_IFAR",
    "current": "current",
    "laser_energy": "laser_energy",
}


def objectives_to_tensor(df, objective_cols=DEFAULT_OBJECTIVE_COLS):
    return torch.tensor(df[objective_cols].to_numpy(dtype=float), **TKWARGS)


def default_ref_point(Y):
    """Reference point 10% below the observed minimum on each objective."""
    y_min = Y.min(dim=0).values
    y_max = Y.max(dim=0).values
    y_range = (y_max - y_min).clamp_min(1e-12)
    return y_min - 0.1 * y_range


def hypervolume_history(df, objective_cols=DEFAULT_OBJECTIVE_COLS, ref_point=None):
    """Cumulative hypervolume after each simulation, in database order.

    Returns ``df`` with two extra columns, ``hypervolume`` (cumulative
    hypervolume using the first ``n`` samples) and ``hv_gain`` (its
    first difference).
    """
    Y = objectives_to_tensor(df, objective_cols)
    if ref_point is None:
        ref_point = default_ref_point(Y)

    hv = []
    for n in range(1, Y.shape[0] + 1):
        bd = DominatedPartitioning(ref_point=ref_point, Y=Y[:n])
        hv.append(bd.compute_hypervolume().item())

    out = df.copy()
    out["hypervolume"] = hv
    out["hv_gain"] = out["hypervolume"].diff()
    out.loc[out.index[0], "hv_gain"] = out["hypervolume"].iloc[0]
    return out


def plot_hypervolume_history(df):
    """Plot cumulative hypervolume history."""
    plt.figure(figsize=(8, 5))
    plt.plot(df["hypervolume"])
    plt.title("Cumulative hypervolume history")
    plt.xlabel("Sample index")
    plt.ylabel("Cumulative hypervolume")
    plt.show()


def plot_correlation_heatmap(df, input_cols=DEFAULT_INPUT_COLS, other_cols=None, objective_cols=DEFAULT_OBJECTIVE_COLS):
    """Pearson correlation heatmap between design inputs and outputs."""
    other_cols = other_cols if other_cols is not None else DEFAULT_OTHER_COLS
    output_cols = other_cols + objective_cols

    corr = df[input_cols + output_cols].corr(method="pearson")
    input_output_corr = corr.loc[input_cols, output_cols]

    plt.figure(figsize=(8, 5))
    im = plt.imshow(input_output_corr, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, label="Pearson correlation")
    plt.xticks(np.arange(len(output_cols)), output_cols, rotation=30, ha="right")
    plt.yticks(np.arange(len(input_cols)), input_cols)

    for i in range(len(input_cols)):
        for j in range(len(output_cols)):
            plt.text(j, i, f"{input_output_corr.iloc[i, j]:.2f}", ha="center", va="center")

    plt.title("Input-output Pearson correlation")
    plt.tight_layout()
    plt.show()
    return input_output_corr


def plot_objective_correlations(
    df,
    input_cols=DEFAULT_INPUT_COLS,
    objective_cols=DEFAULT_OBJECTIVE_COLS,
    objective_labels=DEFAULT_OBJECTIVE_LABELS,
    input_labels=DEFAULT_INPUT_LABELS,
):
    """Rank each input by its absolute Pearson correlation with each objective."""
    objective_corr = df[input_cols + objective_cols].corr().loc[input_cols, objective_cols]

    fig, axes = plt.subplots(1, len(objective_cols), figsize=(14, 4.8), sharex=True)

    for ax, objective in zip(axes, objective_cols):
        values = objective_corr[objective].sort_values(key=np.abs)
        colors = np.where(values >= 0, "#2878B5", "#E4221C")
        bars = ax.barh(
            [input_labels[name] for name in values.index],
            values,
            color=colors,
            alpha=0.9,
        )
        ax.axvline(0, color="0.25", linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(objective_labels[objective])
        ax.set_xlabel("Pearson correlation")
        ax.grid(axis="x", alpha=0.2)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    fig.suptitle("Input relationships with optimisation objectives", fontsize=14)
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="#2878B5", lw=6, label="Positive"),
            plt.Line2D([0], [0], color="#D9534F", lw=6, label="Negative"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.show()
    return objective_corr


def pareto_front(df, objective_cols=DEFAULT_OBJECTIVE_COLS, valid_mask=None):
    """Split ``df`` into the valid subset and its non-dominated (Pareto) rows.

    BoTorch's ``is_non_dominated`` assumes every objective is maximised,
    matching this campaign's ``Y_*``/``Y_minus_*`` convention.
    """
    valid_df = df.copy() if valid_mask is None else df.loc[valid_mask].copy()
    Y = objectives_to_tensor(valid_df, objective_cols)
    mask = is_non_dominated(Y).cpu().numpy()
    pareto_df = valid_df.loc[mask].copy()
    return valid_df, pareto_df


def plot_pareto_3d(valid_df, pareto_df, objective_cols=DEFAULT_OBJECTIVE_COLS[:3], labels=None):
    """3D scatter of all valid simulations with the Pareto front highlighted."""
    labels = labels or [DEFAULT_OBJECTIVE_LABELS.get(c, c) for c in objective_cols]
    x_col, y_col, z_col = objective_cols

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        valid_df[x_col], valid_df[y_col], valid_df[z_col],
        alpha=0.6, s=25, label="All valid simulations",
    )
    ax.scatter(
        pareto_df[x_col], pareto_df[y_col], pareto_df[z_col],
        marker="*", s=100, depthshade=False, label="Pareto-optimal simulations",
    )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title("Observed three-objective Pareto front")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"{len(pareto_df)} Pareto-optimal simulations out of {len(valid_df)} valid simulations.")


def list_dominating_points(
    pareto_df,
    input_cols=DEFAULT_INPUT_COLS,
    objective_cols=DEFAULT_OBJECTIVE_COLS,
    sort_by="Y_TDT",
    ascending=False,
):
    """Compact lookup table for all current Pareto points."""
    lookup = pareto_df.copy()
    lookup.insert(0, "df_index", lookup.index)

    if "index_0" in lookup.columns:
        lookup.insert(1, "simulation_index", lookup["index_0"])
    else:
        lookup.insert(1, "simulation_index", lookup.index)

    shown_cols = [
        "df_index",
        "simulation_index",
        *[c for c in input_cols if c in lookup.columns],
        *[c for c in objective_cols if c in lookup.columns],
    ]

    if sort_by not in lookup.columns:
        raise KeyError(f"sort_by={sort_by!r} is not a column. Choose from: {list(lookup.columns)}")

    return lookup[shown_cols].sort_values(sort_by, ascending=ascending).reset_index(drop=True)


def inspect_dominating_point(
    pareto_df,
    point_index,
    index_type="df_index",
    input_cols=DEFAULT_INPUT_COLS,
    objective_cols=DEFAULT_OBJECTIVE_COLS,
    other_cols=DEFAULT_OTHER_COLS,
    verbose=True,
):
    """Inspect one Pareto/non-dominated simulation.

    Parameters
    ----------
    point_index : int or float
        Index identifying the simulation.
    index_type : {"df_index", "simulation_index", "pareto_position"}
        - "df_index": pandas row index used in the Pareto plot/table.
        - "simulation_index": original database index from index_0.
        - "pareto_position": zero-based row in list_dominating_points().
    """
    valid_index_types = {"df_index", "simulation_index", "pareto_position"}
    if index_type not in valid_index_types:
        raise ValueError(f"index_type must be one of {sorted(valid_index_types)}, not {index_type!r}.")

    if index_type == "df_index":
        if point_index not in pareto_df.index:
            raise KeyError(
                f"DataFrame index {point_index!r} is not on the current Pareto front. "
                "Run list_dominating_points(...) to see valid indices."
            )
        row = pareto_df.loc[point_index]
        df_index = point_index

    elif index_type == "simulation_index":
        if "index_0" not in pareto_df.columns:
            raise KeyError("The dataframe has no index_0 database-index column.")
        matches = pareto_df.index[pareto_df["index_0"] == point_index].tolist()
        if not matches:
            raise KeyError(
                f"Simulation index {point_index!r} is not on the current Pareto front. "
                "Run list_dominating_points(...) to see valid indices."
            )
        if len(matches) > 1:
            raise ValueError(f"Simulation index {point_index!r} appears more than once: {matches}.")
        df_index = matches[0]
        row = pareto_df.loc[df_index]

    else:  # pareto_position
        lookup = list_dominating_points(pareto_df, input_cols=input_cols, objective_cols=objective_cols)
        position = int(point_index)
        if position < 0 or position >= len(lookup):
            raise IndexError(f"pareto_position must be between 0 and {len(lookup) - 1}.")
        df_index = lookup.loc[position, "df_index"]
        row = pareto_df.loc[df_index]

    simulation_index = row["index_0"] if "index_0" in row.index else df_index

    multiplier_inputs = row[[c for c in input_cols if c in row.index]].copy()
    multiplier_inputs.name = "value"

    physical_inputs = scaling.reconstruct_physical_inputs(row)

    objectives = row[[c for c in objective_cols if c in row.index]].copy()
    objectives.name = "value"

    diagnostics = row[[c for c in other_cols if c in row.index]].copy()
    diagnostics.name = "value"

    if verbose:
        print(f"DataFrame index: {df_index}")
        print(f"Database simulation index (index_0): {simulation_index}")
        print("\nOptimisation inputs / multipliers:")
        print(multiplier_inputs.to_frame())
        if physical_inputs is not None:
            print("\nReconstructed physical inputs:")
            print(physical_inputs.to_frame())
        print("\nObjective values:")
        print(objectives.to_frame())
        if not diagnostics.empty:
            print("\nDiagnostic outputs:")
            print(diagnostics.to_frame())

    return {
        "df_index": df_index,
        "simulation_index": simulation_index,
        "row": row,
        "multiplier_inputs": multiplier_inputs,
        "physical_inputs": physical_inputs,
        "objectives": objectives,
        "diagnostics": diagnostics,
    }


def plot_pairwise_objective_fronts(
    valid_df,
    objective_cols=DEFAULT_OBJECTIVE_COLS,
    objective_labels=DEFAULT_OBJECTIVE_LABELS,
):
    """For every pair of objectives, recompute and plot the 2D Pareto front."""
    objective_pairs = list(combinations(objective_cols, 2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for ax, (x_col, y_col) in zip(axes, objective_pairs):
        Y_pair = torch.tensor(valid_df[[x_col, y_col]].to_numpy(dtype=float), dtype=torch.double)
        pair_mask = is_non_dominated(Y_pair).cpu().numpy()
        pair_pareto_df = valid_df.loc[pair_mask].sort_values(x_col)

        ax.scatter(valid_df[x_col], valid_df[y_col], alpha=0.5, s=30, label="All simulations")
        ax.scatter(
            pair_pareto_df[x_col], pair_pareto_df[y_col],
            marker="*", s=100, label="2D Pareto front", zorder=3,
        )

        ax.set_xlabel(objective_labels.get(x_col, x_col))
        ax.set_ylabel(objective_labels.get(y_col, y_col))
        ax.set_title(f"{objective_labels.get(x_col, x_col)} vs {objective_labels.get(y_col, y_col)}")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_dt_ice_thickness_vs_current(df):
    """Reconstruct DT ice thickness for every row and plot it against current."""
    geometry = scaling.dt_ice_geometry_frame(df)
    plot_df = df.assign(DT_ice_thickness_um=geometry["DT_ice_thickness_mm"] * 1e3).sort_values("current")

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df["current"], plot_df["DT_ice_thickness_um"], alpha=0.8)
    plt.xlabel("Current (MA)")
    plt.ylabel(r"DT ice thickness ($\mu$m)")
    plt.title("DT ice-layer thickness vs current")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    pearson_r = plot_df["current"].corr(plot_df["DT_ice_thickness_um"])
    print("Pearson correlation:", pearson_r)
    return pearson_r


# ---------------------------------------------------------------------------
# Pareto-CSV based analysis (dominating_points notebook)
# ---------------------------------------------------------------------------

def prepare_pareto_dataframe(df, column_map=DEFAULT_COLUMN_MAP):
    """Add physical objectives and reconstructed DT-ice quantities to a Pareto table.

    ``df`` is typically the CSV produced by :func:`list_dominating_points`
    (e.g. ``results/pareto_dominating_points.csv``).
    """
    pareto_df = df.copy()

    pareto_df["T_DT"] = pareto_df[column_map["temperature"]]
    pareto_df["rhoR_DT"] = pareto_df[column_map["rhoR"]]
    pareto_df["peak_IFAR"] = -pareto_df[column_map["minus_peak_IFAR"]]
    # NOTE: current_MA reads the raw 'current' column directly. An earlier
    # version derived it as -Y_minus_current, but Y_minus_current is stored
    # as -current/20 (normalised), not -current, which silently made every
    # current_MA value 20x too small.
    pareto_df["current_MA"] = pareto_df[column_map["current"]]

    ice_geometry = scaling.dt_ice_geometry_frame(pareto_df)
    pareto_df = pd.concat([pareto_df, ice_geometry], axis=1)

    reference_inputs = pd.DataFrame(
        {
            "current": pareto_df["current_MA"],
            "f_R_outer": 1.0,
            "f_Pi": 1.0,
            "f_Be": 1.0,
        },
        index=pareto_df.index,
    )
    reference_geometry = scaling.dt_ice_geometry_frame(reference_inputs)

    pareto_df["reference_DT_ice_thickness_mm"] = reference_geometry["DT_ice_thickness_mm"]
    pareto_df["delta_DT_ice_thickness_mm"] = (
        pareto_df["DT_ice_thickness_mm"] - pareto_df["reference_DT_ice_thickness_mm"]
    )

    return pareto_df


_TRADEOFFS = [
    ("rhoR_DT", "T_DT", r"DT fuel $\rho R$", r"DT temperature $T_{DT}$", r"$T_{DT}$ against $\rho R_{DT}$"),
    ("current_MA", "T_DT", "Current [MA]", r"DT temperature $T_{DT}$", r"$T_{DT}$ against current"),
    ("current_MA", "rhoR_DT", "Current [MA]", r"DT fuel $\rho R$", r"$\rho R_{DT}$ against current"),
    ("T_DT", "peak_IFAR", r"DT temperature $T_{DT}$", "Peak IFAR", r"Peak IFAR against $T_{DT}$"),
    ("rhoR_DT", "peak_IFAR", r"DT fuel $\rho R$", "Peak IFAR", r"Peak IFAR against $\rho R_{DT}$"),
    ("current_MA", "peak_IFAR", "Current [MA]", "Peak IFAR", "Peak IFAR against current"),
]

_TRADEOFF_COLOUR_LABELS = {
    "current_MA": "Current [MA]",
    "DT_ice_thickness_mm": "DT ice thickness [mm]",
    "reference_DT_ice_thickness_mm": "Reference DT ice thickness [mm]",
    "delta_DT_ice_thickness_mm": "Ice thickness above reference [mm]",
    "Be_thickness_mm": "Be thickness [mm]",
    "R_outer_mm": "Outer radius [mm]",
}


def plot_pareto_tradeoffs(pareto_df, colour_by="current_MA", point_size=45, alpha=0.8, save_path=None):
    """Plot the six pairwise Pareto trade-offs used in the DT-ice-thickness analysis."""
    required_columns = list(dict.fromkeys(["T_DT", "rhoR_DT", "peak_IFAR", "current_MA", colour_by]))
    missing_columns = [c for c in required_columns if c not in pareto_df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}\nAvailable columns are:\n{pareto_df.columns.tolist()}")

    plot_data = pareto_df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_data.empty:
        raise ValueError(f"No finite rows remain when colouring by '{colour_by}'.")

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(17, 10), constrained_layout=True)
    axes = axes.flatten()

    scatter = None
    for ax, (x_col, y_col, x_label, y_label, title) in zip(axes, _TRADEOFFS):
        scatter = ax.scatter(
            plot_data[x_col], plot_data[y_col], c=plot_data[colour_by],
            cmap="viridis_r", s=point_size, alpha=alpha, edgecolors="black", linewidths=0.4,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    colour_label = _TRADEOFF_COLOUR_LABELS.get(colour_by, colour_by.replace("_", " "))
    colourbar = fig.colorbar(scatter, ax=axes, shrink=0.9)
    colourbar.set_label(colour_label)
    fig.suptitle(f"Pareto trade-offs coloured by {colour_label}", fontsize=16)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_ice_thickness_vs_current(pareto_df, colour_by="f_Be", point_size=60, alpha=0.8, save_path=None):
    """Plot DT ice thickness against drive current and report correlations."""
    required_columns = list(dict.fromkeys(["current_MA", "DT_ice_thickness_mm", colour_by]))
    missing_columns = [c for c in required_columns if c not in pareto_df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}\nAvailable columns are:\n{pareto_df.columns.tolist()}")

    plot_data = (
        pareto_df.loc[pareto_df["ice_geometry_valid"], required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("current_MA")
    )
    if len(plot_data) < 2:
        raise ValueError("At least two valid points are required for the current-ice thickness analysis.")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    scatter = ax.scatter(
        plot_data["current_MA"], plot_data["DT_ice_thickness_mm"], c=plot_data[colour_by],
        cmap="viridis_r", s=point_size, alpha=alpha, edgecolors="black", linewidths=0.4,
        label="Pareto designs",
    )

    current_grid = np.linspace(plot_data["current_MA"].min(), plot_data["current_MA"].max(), 300)

    slope, intercept = np.polyfit(plot_data["current_MA"], plot_data["DT_ice_thickness_mm"], deg=1)
    ax.plot(current_grid, slope * current_grid + intercept, linewidth=2, label="Linear trend through Pareto points")

    colour_label = _TRADEOFF_COLOUR_LABELS.get(colour_by, colour_by.replace("_", " "))
    ax.set_xlabel("Current scale")
    ax.set_ylabel("DT ice thickness [mm]")
    ax.set_title("DT ice thickness against drive current")
    ax.grid(alpha=0.3)
    ax.legend()

    colourbar = fig.colorbar(scatter, ax=ax)
    colourbar.set_label(colour_label)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    pearson_r = plot_data["current_MA"].corr(plot_data["DT_ice_thickness_mm"], method="pearson")
    spearman_r = plot_data["current_MA"].rank().corr(plot_data["DT_ice_thickness_mm"].rank())

    print(f"Number of valid plotted points: {len(plot_data)}")
    print(f"Pearson correlation:  {pearson_r:.3f}")
    print(f"Spearman correlation: {spearman_r:.3f}")
    print(f"Linear trend: DT ice thickness [mm] = {slope:.5f} x current [MA] + {intercept:.5f}")

    return {
        "plot_data": plot_data,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "linear_slope_mm_per_MA": slope,
        "linear_intercept_mm": intercept,
    }
