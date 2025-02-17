import pytest

import gdsfactory as gf


def test_coupler_broadband() -> None:
    cross_section = gf.cross_section.rib()

    gf.components.coupler_broadband(cross_section=cross_section, radius=25)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
