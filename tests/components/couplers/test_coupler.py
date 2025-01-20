import pytest

import gdsfactory as gf


def test_coupler_min_radius() -> None:
    with pytest.raises(ValueError, match="min_bend_radius 1"):
        cross_section = gf.cross_section.strip(radius=1)
        gf.components.coupler(
            cross_section=cross_section, allow_min_radius_violation=False
        )


if __name__ == "__main__":
    test_coupler_min_radius()
