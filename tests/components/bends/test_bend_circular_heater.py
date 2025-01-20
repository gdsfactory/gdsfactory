import pytest

import gdsfactory as gf


def test_bend_circular_heater_min_radius() -> None:
    with pytest.raises(ValueError, match="min_bend_radius 1"):
        gf.components.bend_circular_heater(radius=1, allow_min_radius_violation=False)


if __name__ == "__main__":
    test_bend_circular_heater_min_radius()
