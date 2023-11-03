import numpy as np
import pytest

import gdsfactory as gf
from gdsfactory.component import get_base_components


@pytest.mark.parametrize("num_levels", [1, 2, 3])
@pytest.mark.parametrize("num_poly", [0, 1, 10])
@pytest.mark.parametrize("num_empty", [0, 1, 10])
@pytest.mark.parametrize("allow_empty", [True, False])
def test_get_base_components(
    num_levels: int, num_poly: int, num_empty: int, allow_empty: bool
) -> None:
    components = [gf.Component(f"component_{idx}") for idx in range(num_levels)]
    empties = [gf.Component(f"empty_{idx}") for idx in range(num_empty)]
    circles = [gf.components.circle(radius=r) for r in range(1, num_poly + 1)]

    circle_splits = np.array_split(circles, num_levels)
    empty_splits = np.array_split(empties, num_levels)

    for idx, component in enumerate(components):
        for empty in empty_splits[idx]:
            component.add_ref(empty)
        for circle in circle_splits[idx]:
            component.add_ref(circle)

    if len(components) > 1:
        for c1, c2 in zip(components[-2::-1], components[-1:0:-1], strict=True):
            c1.add_ref(c2)

    base_components = list(get_base_components(components[0], allow_empty))

    num_leaves = num_poly + num_empty + len([c for c in components if not c.references])
    expected = num_leaves if allow_empty else num_poly

    assert len(base_components) == expected
