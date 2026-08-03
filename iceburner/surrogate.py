"""Cleaning a mille-feuille ``State`` and scanning a fitted surrogate.

These helpers operate on any object exposing the mille-feuille
``State``/``Surrogate`` interface (``state.Xs``, ``state.Ys``,
``state.X_names``, ``state.Y_names``, ``surrogate.predict(state, X)``,
...) via duck typing, so this module has no hard dependency on
``millefeuille`` itself -- only numpy/pandas/matplotlib.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import scaling


def _subset_sample_aligned_attribute(object_, attribute_name, mask):
    """Filter ``object_.<attribute_name>`` by ``mask`` if it is sample-aligned."""
    if not hasattr(object_, attribute_name):
        return False

    values = getattr(object_, attribute_name)
    if values is None:
        return False

    try:
        first_dimension = len(values)
    except TypeError:
        return False

    if first_dimension != len(mask):
        print(f"Not filtering {attribute_name}: length {first_dimension} is not {len(mask)}.")
        return False

    row_positions = np.flatnonzero(mask)

    if hasattr(values, "iloc"):
        filtered_values = values.iloc[row_positions].copy()
    else:
        try:
            filtered_values = values[mask]
        except (TypeError, IndexError):
            filtered_values = np.asarray(values)[mask]

    setattr(object_, attribute_name, filtered_values)
    return True


def filter_invalid_simulations(
    state,
    required_objectives=("Y_TDT", "Y_rhoRDT", "Y_minus_peak_IFAR"),
    known_bad_ids=(112, 113, 114, 115),
):
    """Remove failed/numerically corrupted simulations before fitting a surrogate.

    Drops rows that are non-finite, listed in ``known_bad_ids``, or fail a
    loose physical sanity check on ``Y_TDT``/``Y_rhoRDT``/``Y_minus_peak_IFAR``.
    The same mask is applied to every sample-aligned attribute of ``state``
    (``Xs``, ``Ys``, ``Ps``, ``Ss``, ``index``), and ``state.Y_scaler`` is
    refit on the cleaned outputs if present.

    Mutates ``state`` in place (pass a ``copy.deepcopy`` if you need to keep
    the original). Returns ``(state, removed_df)``.
    """
    Y_raw = np.asarray(state.Ys, dtype=float)
    n_raw = len(Y_raw)
    if Y_raw.ndim != 2:
        raise ValueError(f"Expected state.Ys to be two-dimensional, got {Y_raw.shape}.")

    objective_names = list(state.Y_names)
    missing = [n for n in required_objectives if n not in objective_names]
    if missing:
        raise ValueError(f"Missing required objectives: {missing}. Available objectives: {objective_names}")

    tdt_idx = objective_names.index("Y_TDT")
    rhor_idx = objective_names.index("Y_rhoRDT")
    ifar_idx = objective_names.index("Y_minus_peak_IFAR")

    state_index = getattr(state, "index", None)
    candidate_ids = np.asarray(state_index).reshape(-1) if state_index is not None else np.arange(n_raw)
    if len(candidate_ids) != n_raw:
        print("state.index is not sample-aligned; using row positions as sample IDs.")
        candidate_ids = np.arange(n_raw)

    known_bad_mask = np.isin(candidate_ids, np.asarray(known_bad_ids))
    finite_mask = np.isfinite(Y_raw).all(axis=1)

    numerical_failure_mask = (
        (Y_raw[:, tdt_idx] <= 1.0e-12)
        | (Y_raw[:, rhor_idx] <= 0.0)
        | (Y_raw[:, rhor_idx] >= 1.0e6)
        | (Y_raw[:, ifar_idx] >= 0.0)
    )

    valid_mask = finite_mask & ~known_bad_mask & ~numerical_failure_mask
    removed_mask = ~valid_mask

    removed_df = pd.DataFrame(Y_raw[removed_mask], columns=objective_names)
    removed_df.insert(0, "sample_id", candidate_ids[removed_mask])

    print(f"Original samples: {n_raw}")
    print(f"Removed samples:  {removed_mask.sum()}")
    print(f"Retained samples: {valid_mask.sum()}")

    for attribute_name in ["Xs", "Ys", "Ps", "Ss", "index"]:
        _subset_sample_aligned_attribute(state, attribute_name, valid_mask)

    if hasattr(state, "nsamples"):
        try:
            state.nsamples = int(valid_mask.sum())
        except (AttributeError, TypeError):
            pass

    y_scaler = getattr(state, "Y_scaler", None)
    if y_scaler is not None and hasattr(y_scaler, "fit"):
        y_scaler.fit(np.asarray(state.Ys, dtype=float))
        print("Refitted state.Y_scaler on filtered outputs.")
    else:
        print("No refittable state.Y_scaler was found; the surrogate will use the filtered state as supplied.")

    n_filtered = len(state.Ys)
    for attribute_name in ["Xs", "Ys"]:
        values = getattr(state, attribute_name)
        if len(values) != n_filtered:
            raise ValueError(f"{attribute_name} has {len(values)} rows, but state.Ys has {n_filtered}.")

    return state, removed_df


def find_name_index(names, search_text):
    """Find the index of the (first) name containing ``search_text``, case-insensitive."""
    matches = [i for i, name in enumerate(names) if search_text.lower() in str(name).lower()]

    if not matches:
        raise ValueError(f"Could not find '{search_text}'. Available names: {list(names)}")
    if len(matches) > 1:
        print(f"Multiple matches for '{search_text}': {[names[i] for i in matches]}. Using {names[matches[0]]}.")

    return matches[0]


def make_anchor(X_train):
    """A valid central design point (nan-aware median) from the training data."""
    return np.nanmedian(X_train, axis=0)


def unpack_prediction(prediction, n_points, n_outputs):
    """Normalise a ``surrogate.predict(...)`` result to ``(n_points, n_outputs)`` mean/std."""
    if not isinstance(prediction, dict):
        raise TypeError(f"Expected surrogate.predict(...) to return a dictionary, but received {type(prediction)}.")

    Y_mean = np.asarray(prediction["mean"], dtype=float)
    Y_std = np.asarray(prediction.get("std", np.zeros_like(Y_mean)), dtype=float)

    expected = (n_points, n_outputs)
    transposed = (n_outputs, n_points)

    if Y_mean.shape == transposed:
        Y_mean = Y_mean.T
    if Y_std.shape == transposed:
        Y_std = Y_std.T

    Y_mean = np.squeeze(Y_mean)
    Y_std = np.squeeze(Y_std)

    if Y_mean.shape != expected:
        raise ValueError(f"Mean prediction shape is {Y_mean.shape}; expected {expected}.")
    if Y_std.shape != expected:
        raise ValueError(f"Standard-deviation shape is {Y_std.shape}; expected {expected}.")

    return Y_mean, Y_std


def plot_objective_predictions(x_values, Y_mean, Y_std, xlabel, title_prefix, objective_names):
    """Plot surrogate mean +/- std for every objective against a scanned input/derived value."""
    n_objectives = len(objective_names)
    fig, axes = plt.subplots(n_objectives, 1, figsize=(8, 3.8 * n_objectives), squeeze=False)
    axes = axes[:, 0]

    for j, name in enumerate(objective_names):
        mean = Y_mean[:, j]
        std = Y_std[:, j]
        finite = np.isfinite(x_values) & np.isfinite(mean) & np.isfinite(std)

        print(f"{name}: {finite.sum()}/{len(x_values)} finite plotted points")

        if finite.sum() == 0:
            axes[j].text(0.5, 0.5, "No finite predictions", ha="center", va="center", transform=axes[j].transAxes)
        else:
            order = np.argsort(x_values[finite])
            x_plot = x_values[finite][order]
            mean_plot = mean[finite][order]
            std_plot = std[finite][order]

            axes[j].plot(x_plot, mean_plot, linewidth=2, label="Surrogate mean")
            axes[j].fill_between(x_plot, mean_plot - std_plot, mean_plot + std_plot, alpha=0.25, label="RF ensemble spread")

        axes[j].set_xlabel(xlabel)
        axes[j].set_ylabel(str(name))
        axes[j].set_title(f"{title_prefix}: {name}")
        axes[j].grid(alpha=0.3)
        axes[j].legend(loc="best")

    plt.tight_layout()
    plt.show()


def scan_input(
    state,
    surrogate,
    input_search_text,
    X_train=None,
    scan_min=None,
    scan_max=None,
    n_points=100,
    anchor=None,
    xlabel=None,
):
    """Sweep one surrogate input across its data range, holding others at the anchor design."""
    X_train = X_train if X_train is not None else np.asarray(state.Xs, dtype=float)
    input_index = find_name_index(state.X_names, input_search_text)
    input_name = state.X_names[input_index]

    anchor = np.asarray(anchor if anchor is not None else make_anchor(X_train), dtype=float).copy()

    scan_min = np.nanpercentile(X_train[:, input_index], 5) if scan_min is None else scan_min
    scan_max = np.nanpercentile(X_train[:, input_index], 95) if scan_max is None else scan_max
    scan_values = np.linspace(scan_min, scan_max, n_points)

    X_pred = np.repeat(anchor[None, :], n_points, axis=0)
    X_pred[:, input_index] = scan_values

    prediction = surrogate.predict(state, X_pred)
    Y_mean, Y_std = unpack_prediction(prediction, n_points, len(state.Y_names))

    plot_objective_predictions(
        scan_values, Y_mean, Y_std,
        xlabel=xlabel or str(input_name),
        title_prefix=f"Scan of {input_name}",
        objective_names=list(state.Y_names),
    )

    return {
        "input_name": input_name,
        "scan_values": scan_values,
        "X_pred": X_pred,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "anchor": anchor,
    }


def physical_dt_ice_thickness(x_row, state, X_train=None):
    """DT ice thickness (m) for one raw surrogate-input row.

    Uses :func:`iceburner.scaling.dt_ice_geometry`. The drive-current
    entry in ``x_row`` may be stored either as physical MA or as a
    multiplier of the 20 MA reference; this is inferred from the
    training-data range (multiplier columns top out well below 5).
    """
    x_row = np.asarray(x_row, dtype=float)
    X_train = X_train if X_train is not None else np.asarray(state.Xs, dtype=float)

    r_index = find_name_index(state.X_names, "R_outer")
    pi_index = find_name_index(state.X_names, "Pi")
    be_index = find_name_index(state.X_names, "Be")
    current_index = find_name_index(state.X_names, "current")

    training_current = X_train[:, current_index]
    current_value = x_row[current_index]
    current_ma = 20.0 * current_value if np.nanmax(training_current) <= 5 else current_value

    geometry = scaling.dt_ice_geometry(
        current_ma,
        f_R_outer=x_row[r_index],
        f_Pi=x_row[pi_index],
        f_Be=x_row[be_index],
    )
    return geometry["DT_ice_thickness_m"] if geometry["ice_geometry_valid"] else np.nan


def scan_dt_ice_thickness(state, surrogate, X_train=None, pi_min=0.9, pi_max=1.1, n_points=100, anchor=None):
    """Scan ``f_Pi``, convert to physical DT ice thickness, and plot every objective against it."""
    X_train = X_train if X_train is not None else np.asarray(state.Xs, dtype=float)
    pi_index = find_name_index(state.X_names, "Pi")

    anchor = np.asarray(anchor if anchor is not None else make_anchor(X_train), dtype=float).copy()
    pi_scan = np.linspace(pi_min, pi_max, n_points)

    X_pred = np.repeat(anchor[None, :], n_points, axis=0)
    X_pred[:, pi_index] = pi_scan

    ice_thickness_mm = 1e3 * np.array(
        [physical_dt_ice_thickness(row, state, X_train=X_train) for row in X_pred]
    )
    valid_geometry = np.isfinite(ice_thickness_mm)
    print(f"Valid ice geometries: {valid_geometry.sum()}/{n_points}")

    prediction = surrogate.predict(state, X_pred)
    Y_mean, Y_std = unpack_prediction(prediction, n_points, len(state.Y_names))

    x_for_plot = ice_thickness_mm.copy()
    x_for_plot[~valid_geometry] = np.nan

    plot_objective_predictions(
        x_for_plot, Y_mean, Y_std,
        xlabel="DT ice thickness [mm]",
        title_prefix="DT ice-thickness scan",
        objective_names=list(state.Y_names),
    )

    return {
        "pi_scan": pi_scan,
        "ice_thickness_mm": ice_thickness_mm,
        "valid_geometry": valid_geometry,
        "X_pred": X_pred,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "anchor": anchor,
    }
