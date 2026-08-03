"""Similarity-scaling physics shared by every notebook in this project.

The MOBO campaign expresses each design as a set of dimensionless
multipliers (``f_R_outer``, ``f_Pi``, ``f_Be``, ``f_laser``, ``f_rho``)
applied on top of a similarity-scaled reference design computed from the
drive current. This module is the single source of truth for that
scaling relation and for reconstructing the physical DT ice-layer
geometry from it.

All functions accept either plain floats or numpy/pandas array-likes for
their inputs, since the same formulas are used both for one-off point
inspection and for vectorised operations over a whole dataframe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reference design at the baseline drive current.
REFERENCE_CURRENT_MA = 20.0

R_OUTER_20 = 2.79e-3       # m
R_INNER_20 = 2.325e-3      # m
PI_20 = 1.20e-7            # kg/m
LASER_ENERGY_20 = 0.84e3   # J
RHO_GAS_20 = 2.25          # kg/m^3

# Similarity-scaling exponents.
ALPHA_R_OUTER = 0.381
ALPHA_R_INNER = 0.206
ALPHA_PI = 2.0
ALPHA_LASER = 2.529
ALPHA_RHO_GAS = 0.529

# Material densities used to reconstruct the DT ice layer.
RHO_BE = 1.85e3       # kg/m^3
RHO_DT_ICE = 0.92e3   # kg/m^3


def similarity_scale(current_MA):
    """Return s = current / REFERENCE_CURRENT_MA."""
    return np.asarray(current_MA, dtype=float) / REFERENCE_CURRENT_MA


def scaled_reference_values(current_MA):
    """Similarity-scaled reference design at the given drive current(s)."""
    s = similarity_scale(current_MA)

    with np.errstate(invalid="ignore"):
        r_outer_sim = R_OUTER_20 * s**ALPHA_R_OUTER
        r_inner_sim = R_INNER_20 * s**ALPHA_R_INNER
        pi_parameter_sim = PI_20 * s**ALPHA_PI
        laser_energy_sim = LASER_ENERGY_20 * s**ALPHA_LASER
        rho_gas_sim = RHO_GAS_20 * s**ALPHA_RHO_GAS

    return {
        "R_outer_sim": r_outer_sim,
        "R_inner_sim": r_inner_sim,
        "Be_thickness_sim": r_outer_sim - r_inner_sim,
        "Pi_parameter_sim": pi_parameter_sim,
        "laser_energy_sim": laser_energy_sim,
        "rho_gas_sim": rho_gas_sim,
    }


def get_scaled_inputs(
    current_MA,
    f_R_outer=1.0,
    f_Pi=1.0,
    f_Be=1.0,
    f_laser=1.0,
    f_rho=1.0,
):
    """Convert MOBO multipliers into dimensional design inputs.

    Parameters
    ----------
    current_MA : float or array-like
        Drive current in MA.
    f_R_outer, f_Pi, f_Be, f_laser, f_rho : float or array-like
        Multipliers applied to the similarity-scaled reference design.

    Returns
    -------
    dict
        Physical design values (``R_outer``, ``Pi_parameter``,
        ``Be_thickness``, ``laser_energy``, ``rho_gas``,
        ``inner_Be_radius``) plus the underlying reference values used
        to compute them (``*_sim``).
    """
    reference = scaled_reference_values(current_MA)

    R_outer = f_R_outer * reference["R_outer_sim"]
    Pi_parameter = f_Pi * reference["Pi_parameter_sim"]
    Be_thickness = f_Be * reference["Be_thickness_sim"]
    laser_energy = f_laser * reference["laser_energy_sim"]
    rho_gas = f_rho * reference["rho_gas_sim"]

    return {
        "R_outer": R_outer,
        "Pi_parameter": Pi_parameter,
        "Be_thickness": Be_thickness,
        "laser_energy": laser_energy,
        "rho_gas": rho_gas,
        "inner_Be_radius": R_outer - Be_thickness,
        **reference,
    }


def dt_ice_geometry(current_MA, f_R_outer=1.0, f_Pi=1.0, f_Be=1.0):
    """Reconstruct the physical Be/DT-ice geometry from MOBO multipliers.

    Vectorised: every argument may be a scalar or an array-like of the
    same length. Returns a dict of numpy arrays (or 0-d values for
    scalar input) with a boolean ``valid`` mask flagging geometries
    that are not physically realisable (e.g. negative ice thickness).
    """
    current_MA = np.asarray(current_MA, dtype=float)
    f_R_outer = np.asarray(f_R_outer, dtype=float)
    f_Pi = np.asarray(f_Pi, dtype=float)
    f_Be = np.asarray(f_Be, dtype=float)

    reference = scaled_reference_values(current_MA)

    R_outer = f_R_outer * reference["R_outer_sim"]
    Be_thickness = f_Be * reference["Be_thickness_sim"]
    Pi_parameter = f_Pi * reference["Pi_parameter_sim"]
    R_Be_inner = R_outer - Be_thickness

    shell_mass_per_length = Pi_parameter / R_outer**2
    Be_mass_per_length = np.pi * (R_outer**2 - R_Be_inner**2) * RHO_BE
    DT_ice_mass_per_length = shell_mass_per_length - Be_mass_per_length

    with np.errstate(invalid="ignore"):
        ice_radius_squared = R_Be_inner**2 - DT_ice_mass_per_length / (
            np.pi * RHO_DT_ICE
        )

    valid = (
        np.isfinite(current_MA)
        & (current_MA > 0)
        & (R_outer > 0)
        & (Be_thickness > 0)
        & (R_Be_inner > 0)
        & (DT_ice_mass_per_length > 0)
        & (ice_radius_squared >= 0)
    )

    R_DT_ice_inner = np.full_like(np.atleast_1d(R_Be_inner), np.nan, dtype=float)
    valid_flat = np.atleast_1d(valid)
    R_DT_ice_inner[valid_flat] = np.sqrt(
        np.atleast_1d(ice_radius_squared)[valid_flat]
    )
    if np.ndim(R_Be_inner) == 0:
        R_DT_ice_inner = R_DT_ice_inner[0]
        valid = bool(valid)

    DT_ice_thickness = R_Be_inner - R_DT_ice_inner

    geometry = {
        "R_outer_m": R_outer,
        "Be_thickness_m": Be_thickness,
        "R_Be_inner_m": R_Be_inner,
        "R_DT_ice_inner_m": R_DT_ice_inner,
        "DT_ice_thickness_m": DT_ice_thickness,
        "shell_mass_per_length": shell_mass_per_length,
        "Be_mass_per_length": Be_mass_per_length,
        "DT_ice_mass_per_length": DT_ice_mass_per_length,
        "ice_geometry_valid": valid,
    }

    for key in ("R_outer", "Be_thickness", "R_Be_inner", "R_DT_ice_inner", "DT_ice_thickness"):
        geometry[f"{key}_mm"] = geometry[f"{key}_m"] * 1e3

    return geometry


def dt_ice_geometry_frame(data):
    """DataFrame convenience wrapper around :func:`dt_ice_geometry`.

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain columns ``f_R_outer``, ``f_Pi``, ``f_Be`` and
        either ``current`` or ``current_MA``.

    Returns
    -------
    pandas.DataFrame
        Same index as ``data``, with the geometry fields described in
        :func:`dt_ice_geometry`.
    """
    required_columns = ["f_R_outer", "f_Pi", "f_Be"]
    missing_columns = [c for c in required_columns if c not in data.columns]
    if missing_columns:
        raise KeyError(
            f"Missing columns needed for the DT ice calculation: {missing_columns}"
        )

    if "current" in data.columns:
        current_MA = pd.to_numeric(data["current"], errors="coerce")
    elif "current_MA" in data.columns:
        current_MA = pd.to_numeric(data["current_MA"], errors="coerce")
    else:
        raise KeyError("The dataframe must contain either 'current' or 'current_MA'.")

    geometry = dt_ice_geometry(
        current_MA.to_numpy(),
        f_R_outer=pd.to_numeric(data["f_R_outer"], errors="coerce").to_numpy(),
        f_Pi=pd.to_numeric(data["f_Pi"], errors="coerce").to_numpy(),
        f_Be=pd.to_numeric(data["f_Be"], errors="coerce").to_numpy(),
    )
    return pd.DataFrame(geometry, index=data.index)


def reconstruct_physical_inputs(row):
    """Convert one Pareto-point row's multipliers into physical inputs.

    Parameters
    ----------
    row : pandas.Series or Mapping
        Must contain ``f_laser``, ``f_R_outer``, ``f_Pi``, ``f_Be``,
        ``f_rho`` and ``current``.

    Returns
    -------
    pandas.Series or None
        Physical values (``laser_energy_J``, ``R_outer_m``,
        ``Pi_parameter_kg_per_m``, ``Be_thickness_m``,
        ``rho_gas_kg_per_m3``, ``current_MA``), or ``None`` if the row
        is missing any required field.
    """
    required = {"f_laser", "f_R_outer", "f_Pi", "f_Be", "f_rho", "current"}
    if not required.issubset(row.keys() if hasattr(row, "keys") else row.index):
        return None

    physical = get_scaled_inputs(
        float(row["current"]),
        f_R_outer=float(row["f_R_outer"]),
        f_Pi=float(row["f_Pi"]),
        f_Be=float(row["f_Be"]),
        f_laser=float(row["f_laser"]),
        f_rho=float(row["f_rho"]),
    )

    return pd.Series(
        {
            "laser_energy_J": physical["laser_energy"],
            "R_outer_m": physical["R_outer"],
            "Pi_parameter_kg_per_m": physical["Pi_parameter"],
            "Be_thickness_m": physical["Be_thickness"],
            "rho_gas_kg_per_m3": physical["rho_gas"],
            "current_MA": float(row["current"]),
        },
        name="physical_value",
    )
