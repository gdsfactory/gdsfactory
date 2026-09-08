import pytest

import gdsfactory as gf


def test_coupler_min_radius() -> None:
    with pytest.raises(ValueError, match="min_bend_radius 1"):
        gf.components.coupler(
            cross_section="strip", radius=1, allow_min_radius_violation=False
        )
    gf.components.coupler(
        cross_section="strip", radius=10, allow_min_radius_violation=True
    )
