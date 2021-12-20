"""pack a list of components into a grid """
from typing import Optional, Tuple

import numpy as np
from phidl.device_layout import Group

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.difftest import difftest
from gdsfactory.types import Anchor, ComponentFactory, ComponentOrFactory, Float2


@cell
def grid(
    components: Tuple[ComponentOrFactory, ...],
    spacing: Tuple[float, float] = (5.0, 5.0),
    separation: bool = True,
    shape: Tuple[int, int] = None,
    align_x: str = "x",
    align_y: str = "y",
    edge_x: str = "x",
    edge_y: str = "ymax",
    rotation: int = 0,
    h_mirror: bool = False,
    v_mirror: bool = False,
) -> Component:
    """Returns a component with a 1D or 2D grid of components

    Adapted from phidl.geometry

    Args:
        components: Iterable to be placed onto a grid. (can be 1D or 2D)
        spacing: between adjacent elements on the grid, can be a tuple for
            different distances in height and width.
        separation: If True, guarantees elements are speparated with fixed spacing
            if False, elements are spaced evenly along a grid.
        shape: x, y shape of the grid (see np.reshape).
            If no shape and the list is 1D, if np.reshape were run with (1, -1).
        align_x: {'x', 'xmin', 'xmax'} for x (column) alignment along
        align_y: {'y', 'ymin', 'ymax'} for y (row) alignment along
        edge_x: {'x', 'xmin', 'xmax'} for x (column) distribution (ignored if separation = True)
        edge_y: {'y', 'ymin', 'ymax'} for y (row) distribution along (ignored if separation = True)
        rotation: for each component in degrees
        h_mirror: horizontal mirror using y axis (x, 1) (1, 0). This is the most common mirror.
        v_mirror: vertical mirror using x axis (1, y) (0, y)

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
            d = d() if callable(d) else d
            ref = d.ref(rotation=rotation, h_mirror=h_mirror, v_mirror=v_mirror)
            D.add(ref)
            ref_array[idx] = ref

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


@cell
def grid_with_text(
    components: Tuple[ComponentOrFactory, ...],
    text_prefix: str = "",
    text_offsets: Tuple[Float2, ...] = ((0, 0),),
    text_anchors: Tuple[Anchor, ...] = ("cc",),
    text: Optional[ComponentFactory] = text_rectangular,
    **kwargs,
) -> Component:
    """Returns Grid with text labels

    Args:
        components: Iterable to be placed onto a grid. (can be 1D or 2D)
        text_prefix: for labels. For example. 'A' will produce 'A1', 'A2', ...
        text_offsets: relative to component anchor. Defaults to center
        text_anchors: relative to component (ce cw nc ne nw sc se sw center cc)
        text: function to add text labels.

    keyword Args:
        spacing: between adjacent elements on the grid, can be a tuple for
          different distances in height and width.
        separation: If True, guarantees elements are speparated with fixed spacing
          if False, elements are spaced evenly along a grid.
        shape: x, y shape of the grid (see np.reshape).
          If no shape and the list is 1D, if np.reshape were run with (1, -1).
        align_x: {'x', 'xmin', 'xmax'}
          to perform the x (column) alignment along
        align_y: {'y', 'ymin', 'ymax'}
          to perform the y (row) alignment along
        edge_x: {'x', 'xmin', 'xmax'}
          to perform the x (column) distribution (ignored if separation = True)
        edge_y: {'y', 'ymin', 'ymax'}
          to perform the y (row) distribution along (ignored if separation = True)
        rotation: for each reference in degrees

    """
    c = Component()
    g = grid(components=components, **kwargs)
    c << g
    if text:
        for i, ref in enumerate(g.aliases.values()):
            for text_offset, text_anchor in zip(text_offsets, text_anchors):
                t = c << text(f"{text_prefix}{i}")
                t.move((np.array(text_offset) + getattr(ref.size_info, text_anchor)))
    return c


def test_grid():
    import gdsfactory as gf

    components = [gf.components.rectangle(size=(i, i)) for i in range(1, 10)]
    c = grid(components)
    difftest(c)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # components = [gf.components.rectangle(size=(i, i)) for i in range(40, 66, 5)]
    # components = [gf.components.rectangle(size=(i, i)) for i in range(40, 66, 5)]

    c = [gf.components.triangle(x=i) for i in range(1, 10)]
    c = grid_with_text(
        c,
        shape=(1, len(c)),
        rotation=0,
        h_mirror=False,
        v_mirror=True,
        spacing=(100, 100),
        text_offsets=((0, 100), (0, -100)),
        text_anchors=("nc", "sc"),
    )
    c.show()
