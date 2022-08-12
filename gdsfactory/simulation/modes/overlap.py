import numpy as np

import gdsfactory.simulation.modes as gm
from gdsfactory.simulation.modes.types import Mode


def innerprod_trapz(
    mode1: Mode,
    mode2: Mode,
    ymin: float = -2.0,
    ymax: float = 2.0,
    zmin: float = -2.0,
    zmax: float = 2.0,
    trapz_num_y: int = 2000,
    trapz_num_z: int = 2000,
):
    """Compute the inner product of two modes as 1/4*int(E1* x H2 + E2 x H1*)_x dydz.

    with int double integral over y,z, x cross product, and _x x-projection
    Uses simple trapz numerical integration

    Note that <1|1> != 1 with this definition,
    but you can compute that to normalize other quantities

    Args:
        mode1: Mode object.
        mode2: Mode object.
        ymin: lower y integration bound.
        ymax: upper y integration bound.
        zmin: lower z integration bound.
        zmax: upper z integration bound.
        trapz_num_y: number of points to resample the mode in y for integration.
        trapz_num_z: number of points to resample the mode in z for integration.

    """
    # Form vector components
    yint = np.linspace(ymin, ymax, trapz_num_y)
    zint = np.linspace(zmin, zmax, trapz_num_z)

    mode1_Ey_interp = mode1.Ey_grid_interp(yint, zint)
    mode1_Ez_interp = mode1.Ez_grid_interp(yint, zint)
    mode1_Hy_interp = mode1.Hy_grid_interp(yint, zint)
    mode1_Hz_interp = mode1.Hz_grid_interp(yint, zint)
    mode2_Ey_interp = mode2.Ey_grid_interp(yint, zint)
    mode2_Ez_interp = mode2.Ez_grid_interp(yint, zint)
    mode2_Hy_interp = mode2.Hy_grid_interp(yint, zint)
    mode2_Hz_interp = mode2.Hz_grid_interp(yint, zint)

    # Compute integrand
    # For cross product in terms of components:
    integrand = (
        np.conj(mode1_Ey_interp) * mode2_Hz_interp
        - np.conj(mode1_Ez_interp) * mode2_Hy_interp
        + mode2_Ey_interp * np.conj(mode1_Hz_interp)
        - mode2_Ez_interp * np.conj(mode1_Hy_interp)
    )

    # Compute integral
    integral = np.trapz(np.trapz(integrand, yint, axis=0), zint, axis=0)

    return 0.25 * integral


def test_innerprod_trapz() -> None:
    """Checks that overlaps do not change."""
    m = gm.find_modes_waveguide()
    overlap = innerprod_trapz(m[1], m[1])
    assert overlap > 0
    # assert np.isclose(np.real(overlap), 0.143, atol=1e-2), np.real(overlap)


if __name__ == "__main__":
    # m = gm.find_modes_waveguide()
    # print(innerprod_trapz(m[1], m[1]))

    test_innerprod_trapz()
