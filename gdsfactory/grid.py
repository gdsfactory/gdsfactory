"""From phidl

"""
from typing import Tuple

import numpy as np
from phidl.device_layout import Group

from gdsfactory.component import Component
from gdsfactory.difftest import difftest


def grid(
    components: Tuple[Component, ...],
    spacing: Tuple[float, float] = (5.0, 5.0),
    separation: bool = True,
    shape: Tuple[int, int] = None,
    align_x: str = "x",
    align_y: str = "y",
    edge_x: str = "x",
    edge_y: str = "ymax",
) -> Component:
    """Places the devices in the `components` (1D or 2D) on a grid.

    Adapted from phid.geometry

    Args:
        components: Iterable to be placed onto a grid.
        spacing: between adjacent elements on the grid, can be a tuple for
            different distances in height and width.
        separation: If True, guarantees elements are speparated with a fixed spacing between;
            if False, elements are spaced evenly along a grid.
        shape: x, y shape of the grid (see np.reshape).
            If no shape is given and the list is 1D, the output is as if np.reshape were run with (1, -1).
        align_x: {'x', 'xmin', 'xmax'}
            Which edge to perform the x (column) alignment along
        align_y: {'y', 'ymin', 'ymax'}
            Which edge to perform the y (row) alignment along
        edge_x: {'x', 'xmin', 'xmax'}
            Which edge to perform the x (column) distribution along (unused if
            separation == True)
        edge_y: {'y', 'ymin', 'ymax'}
            Which edge to perform the y (row) distribution along (unused if
            separation == True)

    Returns:
        Component containing all the components in a grid.
    """

    device_array = np.asarray(components)
    # Check arguments
    if device_array.ndim not in (1, 2):
        raise ValueError("[PHIDL] grid() The components needs to be 1D or 2D.")
    if shape is not None and len(shape) != 2:
        raise ValueError(
            "[PHIDL] grid() shape argument must be None or"
            + " have a length of 2, for example shape=(4,6)"
        )

    # Check that shape is valid and reshape array if needed
    if (shape is None) and (device_array.ndim == 2):  # Already in desired shape
        shape = device_array.shape
    elif (shape is None) and (device_array.ndim == 1):
        shape = (device_array.size, -1)
    elif 0 < shape[0] * shape[1] < device_array.size:
        raise ValueError(
            "[PHIDL] grid() The shape is too small for all the items in components"
        )
    else:
        if np.min(shape) == -1:
            remainder = np.max(shape) - device_array.size % np.max(shape)
        else:
            remainder = shape[0] * shape[1] - device_array.size
        if remainder != 0:
            device_array = np.append(
                device_array,
                [
                    None,
                ]
                * remainder,
            )
    device_array = np.reshape(device_array, shape)

    D = Component("grid")
    ref_array = np.empty(device_array.shape, dtype=object)
    dummy = Component()
    for idx, d in np.ndenumerate(device_array):
        if d is not None:
            ref_array[idx] = D << d
        else:
            ref_array[idx] = D << dummy  # Create dummy devices
        D.aliases[idx] = ref_array[idx]

    rows = [Group(ref_array[n, :]) for n in range(ref_array.shape[0])]
    cols = [Group(ref_array[:, n]) for n in range(ref_array.shape[1])]

    # Align rows and columns independently
    for r in rows:
        r.align(alignment=align_y)
    for c in cols:
        c.align(alignment=align_x)

    # Distribute rows and columns
    Group(cols).distribute(
        direction="x", spacing=spacing[0], separation=separation, edge=edge_x
    )
    Group(rows[::-1]).distribute(
        direction="y", spacing=spacing[1], separation=separation, edge=edge_y
    )

    return D


def test_grid():
    import gdsfactory as gf

    components = [gf.components.rectangle(size=(i, i)) for i in range(1, 10)]
    c = grid(components)
    difftest(c)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    components = [gf.components.rectangle(size=(i, i)) for i in range(1, 10)]
    c = grid(components)
    c.show()
