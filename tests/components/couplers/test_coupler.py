import pytest

import gdsfactory as gf


def test_coupler_min_radius() -> None:
    cross_section = gf.cross_section.strip(radius=1)

    with pytest.raises(ValueError, match="min_bend_radius 1"):
        gf.components.coupler(
            cross_section=cross_section, allow_min_radius_violation=False
        )
    cross_section = gf.cross_section.strip(radius=10)
    gf.components.coupler(cross_section=cross_section, allow_min_radius_violation=True)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
