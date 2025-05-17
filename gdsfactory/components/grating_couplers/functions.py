from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from numpy import pi, sin, sqrt

import gdsfactory as gf
from gdsfactory.functions import DEG2RAD, extrude_path

neff_ridge = 2.8
neff_shallow = 2.5


def ellipse_arc(
    a: float,
    b: float,
    x0: float,
    theta_min: float,
    theta_max: float,
    angle_step: float = 0.5,
) -> npt.NDArray[np.floating[Any]]:
    """Returns an elliptical arc.

    b = a *sqrt(1-e**2)

    An ellipse with a = b has zero eccentricity (is a circle)

    Args:
        a: ellipse semi-major axis.
        b: semi-minor axis.
        x0: in um.
        theta_min: in rad.
        theta_max: in rad.
        angle_step: in rad.
    """
    # Make use of STEP and limits consistent with inclusion at max; avoid repeated allocation in snap
    theta = (
        np.arange(theta_min, theta_max + angle_step, angle_step, dtype=np.float64)
        * DEG2RAD
    )

    # Compute xs, ys as float64 and create result array in one step
    xs = a * np.cos(theta) + x0
    ys = b * np.sin(theta)

    # Preallocate and fill in one pass, avoid extra temp arrays/object construction
    # (stacking = 1 array allocation, 2 assignments)
    arc = np.empty((xs.size, 2), dtype=np.float64)
    arc[:, 0] = xs
    arc[:, 1] = ys

    # Use internal fast snap_to_grid for 2d arrays
    arc = _fast_snap_to_grid_2d(arc)
    return arc  # shape (N,2), as before


def grating_tooth_points(
    ap: float,
    bp: float,
    xp: float,
    width: float,
    taper_angle: float,
    spiked: bool = True,
    angle_step: float = 1.0,
) -> npt.NDArray[np.floating[Any]]:
    theta_min = -taper_angle / 2
    theta_max = taper_angle / 2

    backbone_points = ellipse_arc(ap, bp, xp, theta_min, theta_max, angle_step)
    spike_length = width / 3 if spiked else 0.0
    return extrude_path(
        backbone_points,
        width,
        with_manhattan_facing_angles=False,
        spike_length=spike_length,
    )


def grating_taper_points(
    a: float,
    b: float,
    x0: float,
    taper_length: float,
    taper_angle: float,
    wg_width: float,
    angle_step: float = 1.0,
) -> npt.NDArray[np.floating[Any]]:
    """Returns a taper for a grating coupler.

    Args:
        a: ellipse semi-major axis.
        b: semi-minor axis.
        x0: in um.
        taper_length: in um.
        taper_angle: in degrees.
        wg_width: in um.
        angle_step: in degrees.
    """
    # ellipse_arc returns already snapped, optimal points
    taper_arc = ellipse_arc(
        a=a,
        b=b,
        x0=taper_length,
        theta_min=-taper_angle / 2,
        theta_max=taper_angle / 2,
        angle_step=angle_step,
    )

    # Create both "end" points and write directly to output array, with float64 precision
    n_arc = taper_arc.shape[0]
    out = np.empty((n_arc + 2, 2), dtype=np.float64)
    out[0, 0] = x0
    out[0, 1] = wg_width * 0.5
    out[1, 0] = x0
    out[1, 1] = -wg_width * 0.5
    out[2:] = taper_arc
    return out


def get_grating_period_curved(
    fiber_angle: float = 15.0,
    wavelength: float = 1.55,
    n_slab: float = (neff_ridge + neff_shallow) / 2,
    n_clad: float = 1.0,
) -> tuple[float, float]:
    """The following function calculates the confocal grating periods n_slab is.

    the "average slab index" of the grating. For 220nm silicon it is 2.8, for
    150nm it is 2.5. The average is approximately 2.65. n_clad is the cladding
    index in which the fiber is located, not the index of the layer above the
    straight. If the fiber is in air, then it is 1.0. If you use an index
    matching fluid or glue, then it should be 1.45.

    Args:
        fiber_angle: in degrees.
        wavelength: um.
        n_slab: slab refractive index.
        n_clad: cladding refractive index.
    """
    DEG2RAD = pi / 180
    cos_fib_angle = sin(DEG2RAD * fiber_angle)
    n2_reduced = n_slab**2 - n_clad**2 * cos_fib_angle**2
    sqrt_n2_reduced = sqrt(n2_reduced)
    h_period = wavelength * (n_slab + n_clad * cos_fib_angle) / n2_reduced
    v_period = wavelength / sqrt_n2_reduced
    return h_period, v_period


def get_grating_period(
    fiber_angle: float = 13.45,
    wavelength: float = 1.55,
    neff_high: float = neff_ridge,
    neff_low: float = neff_shallow,
    n_clad: float = 1.45,
) -> float:
    """Return grating coupler period based on lumerical slides.

    Args:
        fiber_angle: in degrees.
        wavelength: um.
        neff_high: high index.
        neff_low: low index.
        n_clad: cladding index.
    """
    DEG2RAD = pi / 180
    neff = (neff_high + neff_low) / 2
    return wavelength / (neff - float(sin(DEG2RAD * fiber_angle)) * n_clad)


# Cythonize-compatible fast vectorized snapping for 2D arrays
def _fast_snap_to_grid_2d(
    arr: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """A faster 2D snap-to-grid specifically tuned for ellipse points (avoids input checks and asarray).
    Assumes arr is float64 numpy array of shape (N,2).
    """
    # Equivalent to gf.snap.snap_to_grid(arr) but avoids repeated type-checking,
    # conversion, and temporary allocations.
    grid_size = gf.snap.kf.kcl.dbu
    nm = round(grid_size * 1000)
    # Direct NumPy math, no type coercions or dtype guessing
    scale = 1e3 / nm
    # Multiply by scale, round, multiply by nm, divide by 1e3 (single operation for both columns)
    np.multiply(arr, scale, out=arr)
    np.rint(arr, out=arr)
    np.multiply(arr, nm / 1e3, out=arr)
    return arr


if __name__ == "__main__":
    pc = get_grating_period_curved()
    p = get_grating_period()
    print(pc)
    print(p)
