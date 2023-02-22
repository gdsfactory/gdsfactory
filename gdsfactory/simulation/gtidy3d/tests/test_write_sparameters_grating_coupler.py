from __future__ import annotations

import numpy as np

import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt

fiber_port_name = "o2"


def test_sparameters_grating_coupler(overwrite=True) -> None:
    """Checks Sparameters for a grating_coupler_elliptical_arbitrary in 2D."""
    c = gf.components.grating_coupler_elliptical_arbitrary(
        widths=[0.343] * 25, gaps=[0.345] * 25
    )

    fiber_angle_deg = 20
    offsets = [0]
    dfs = [
        gt.write_sparameters_grating_coupler(
            component=c,
            is_3d=False,
            fiber_angle_deg=fiber_angle_deg,
            fiber_xoffset=fiber_xoffset,
            overwrite=overwrite,
        )
        for fiber_xoffset in offsets
    ]
    sp = dfs[0]

    # Check reasonable reflection/transmission
    transmission = np.abs(sp[f"{fiber_port_name}@0,o1@0"])
    reflection = np.abs(sp["o1@0,o1@0"])

    assert 1 > transmission.min() > 0.2, transmission.min()
    assert 0.3 > reflection.max() > 0, reflection.max()


if __name__ == "__main__":
    overwrite = True
    c = gf.components.grating_coupler_elliptical_arbitrary(
        widths=[0.343] * 25, gaps=[0.345] * 25
    )

    fiber_angle_deg = 20
    offsets = [0, 1]
    dfs = [
        gt.write_sparameters_grating_coupler(
            component=c,
            is_3d=False,
            fiber_angle_deg=fiber_angle_deg,
            fiber_xoffset=fiber_xoffset,
            overwrite=overwrite,
        )
        for fiber_xoffset in offsets
    ]
    sp = dfs[0]
    transmission = np.abs(sp[f"{fiber_port_name}@0,o1@0"])
    reflection = np.abs(sp["o1@0,o1@0"])

    assert 1 > transmission.min() > 0.2, transmission.min()
    assert 0.3 > reflection.max() > 0, reflection.max()
