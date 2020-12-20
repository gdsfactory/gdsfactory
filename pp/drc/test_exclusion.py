import numpy as np
import pytest

import pp
from pp.drc import check_exclusion


def get_device(space, width=0.5, layer1=(1, 0), layer2=(2, 0)):
    c = pp.Component()
    r1 = c << pp.c.rectangle(size=(width, width), layer=layer1)
    r2 = c << pp.c.rectangle(size=(width, width), layer=layer2)
    r1.xmax = 0
    r2.xmin = space
    return c


@pytest.mark.parametrize(
    "space,min_space,area_expected", [(0.16, 0.1, 0), (0.1, 0.11, 50000)]
)
def test_exclusion(space, min_space, area_expected):
    c = get_device(space=space)
    area = check_exclusion(c, min_space=min_space)
    assert np.isclose(area, area_expected)


if __name__ == "__main__":
    test_exclusion()
