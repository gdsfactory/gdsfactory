from typing import Tuple

import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory.geometry.compute_area import compute_area


@pytest.mark.parametrize("x,y,layer", [(1, 8, (1, 0)), (9, 1, (2, 2))])
def test_compute_area(x: int, y: int, layer: Tuple[int, int]) -> None:
    c = gf.components.rectangle(size=(x, y), layer=layer)
    c.write_gds()
    area = compute_area(c, layer)
    assert np.isclose(area, x * y)


if __name__ == "__main__":
    test_compute_area(1, 2, (1, 0))
