from typing import Tuple

import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.geometry import check_exclusion


@gf.cell
def exclusion(
    space: float,
    width: float = 0.5,
    layer1: Tuple[int, int] = (1, 0),
    layer2: Tuple[int, int] = (2, 0),
) -> Component:
    c = gf.Component()
    r1 = c << gf.components.rectangle(size=(width, width), layer=layer1)
    r2 = c << gf.components.rectangle(size=(width, width), layer=layer2)
    r1.xmax = 0
    r2.xmin = space
    return c


@pytest.mark.parametrize(
    "space,min_space,area_expected", [(0.16, 0.1, 0), (0.1, 0.11, 50000)]
)
def test_exclusion(space: float, min_space: float, area_expected: int) -> None:
    c = exclusion(space=space)
    area = check_exclusion(c, min_space=min_space)
    assert np.isclose(area, area_expected)


if __name__ == "__main__":
    test_exclusion(0.16, 0.1, 0)
