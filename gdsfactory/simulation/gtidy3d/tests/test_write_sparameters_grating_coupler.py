from __future__ import annotations

import numpy as np

import gdsfactory as gf
import gdsfactory.simulation.gtidy3d as gt


def test_sparameters_grating_coupler(overwrite=True) -> None:
    """Checks Sparameters for a grating_coupler_elliptical_arbitrary in 2D."""
    c = gf.components.grating_coupler_elliptical_arbitrary(
        widths=[0.343] * 25, gaps=[0.345] * 25
    )

    sp = gt.write_sparameters_1x1(c, overwrite=overwrite, is_3d=False)

    fiber_angle_deg = 20
    offsets = [0]
    jobs = [
        dict(
            component=c,
            is_3d=False,
            fiber_angle_deg=fiber_angle_deg,
            fiber_xoffset=fiber_xoffset,
        )
        for fiber_xoffset in offsets
    ]
    dfs = gt.write_sparameters_grating_coupler_batch(jobs)
    sp = dfs[0]

    # Check reasonable reflection/transmission
    assert 0 > np.abs(sp["o1@0,o2@0"]).min() > 0.89, np.abs(sp["o1@0,o2@0"]).min()
    assert 0 > np.abs(sp["o1@0,o1@0"]).max() < 0.1, np.abs(sp["o1@0,o1@0"]).max()


if __name__ == "__main__":
    overwrite = False
    c = gf.components.grating_coupler_elliptical_arbitrary(
        widths=[0.343] * 25, gaps=[0.345] * 25
    )

    sp = gt.write_sparameters_1x1(c, overwrite=overwrite, is_3d=False)

    fiber_angle_deg = 20
    offsets = [0]
    jobs = [
        dict(
            component=c,
            is_3d=False,
            fiber_angle_deg=fiber_angle_deg,
            fiber_xoffset=fiber_xoffset,
        )
        for fiber_xoffset in offsets
    ]
    dfs = gt.write_sparameters_grating_coupler_batch(jobs)
