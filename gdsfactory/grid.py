"""pack a list of components into a grid.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""
from __future__ import annotations

import numpy as np

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.component_layout import Group
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.components.triangles import triangle
from gdsfactory.typings import Anchor, ComponentSpec, Float2


@cell
def grid(
    components: tuple[ComponentSpec, ...] | None = None,
    spacing: tuple[float, float] = (5.0, 5.0),
    separation: bool = True,
    shape: tuple[int, int] | None = None,
    align_x: str = "x",
    align_y: str = "y",
    edge_x: str = "x",
    edge_y: str = "ymax",
    rotation: int = 0,
    h_mirror: bool = False,
    v_mirror: bool = False,
    add_ports_prefix: bool = True,
    add_ports_suffix: bool = False,
) -> Component:
    """Returns Component with a 1D or 2D grid of components.

    Args:
        components: Iterable to be placed onto a grid. (can be 1D or 2D).
        spacing: between adjacent elements on the grid, can be a tuple for \
                different distances in height and width.
        separation: If True, guarantees elements are separated with fixed spacing \
                if False, elements are spaced evenly along a grid.
        shape: x, y shape of the grid (see np.reshape). \
                If no shape and the list is 1D, if np.reshape were run with (1, -1).
        align_x: {'x', 'xmin', 'xmax'} for x (column) alignment along.
        align_y: {'y', 'ymin', 'ymax'} for y (row) alignment along.
        edge_x: {'x', 'xmin', 'xmax'} for x (column) (ignored if separation = True).
        edge_y: {'y', 'ymin', 'ymax'} for y (row) along (ignored if separation = True).
        rotation: for each component in degrees.
        h_mirror: horizontal mirror y axis (x, 1) (1, 0). most common mirror.
        v_mirror: vertical mirror using x axis (1, y) (0, y).
        add_ports_prefix: adds port names with prefix.
        add_ports_suffix: adds port names with suffix.

    Returns:
        Component containing components grid.

    .. plot::
        :include-source:

        import gdsfactory as gf

        components = [gf.components.triangle(x=i) for i in range(1, 10)]
        c = gf.grid(
            components,
            shape=(1, len(components)),
            rotation=0,
            h_mirror=False,
            v_mirror=True,
            spacing=(100, 100),
        )
        c.plot()

    """
    components = components or [triangle(x=i) for i in range(1, 10)]
    device_array = np.asarray(components)

    # Check arguments
    if device_array.ndim not in (1, 2):
        raise ValueError("grid() The components needs to be 1D or 2D.")
    if shape is not None and len(shape) != 2:
        raise ValueError(
            "grid() shape argument must be None or"
            f" have a length of 2, for example shape=(4,6), got {shape}"
        )

    # Check that shape is valid and reshape array if needed
    if (shape is None) and (device_array.ndim == 2):  # Already in desired shape
        shape = device_array.shape
    elif (shape is None) and (device_array.ndim == 1):
        shape = (device_array.size, -1)
    elif 0 < shape[0] * shape[1] < device_array.size:
        raise ValueError(
            f"Shape {shape} is too small for all {device_array.size} components"
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
    prefix_to_ref = {}

    D = Component()
    ref_array = np.empty(device_array.shape, dtype=object)
    dummy = Component()
    for idx, d in np.ndenumerate(device_array):
        if d is not None:
            d = d() if callable(d) else d
            ref = D.add_ref(d, rotation=rotation, x_reflection=h_mirror)
            if v_mirror:
                ref.mirror_y()
            ref_array[idx] = ref
            prefix = f"{idx}"
            prefix = prefix.replace(" ", "")
            prefix = prefix.replace(",", "_")
            prefix = prefix.replace("(", "")
            prefix = prefix.replace(")", "")
            ref.name = prefix
            prefix_to_ref[prefix] = ref

        else:
            ref_array[idx] = D << dummy  # Create dummy devices

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

    for prefix, ref in prefix_to_ref.items():
        if add_ports_prefix:
            D.add_ports(ref.ports, prefix=f"{prefix}_")
        elif add_ports_suffix:
            D.add_ports(ref.ports, suffix=f"_{prefix}")
        else:
            D.add_ports(ref.ports)

    return D


@cell
def grid_with_text(
    components: tuple[ComponentSpec, ...] | None = None,
    text_prefix: str = "",
    text_offsets: tuple[Float2, ...] = ((0, 0),),
    text_anchors: tuple[Anchor, ...] = ("cc",),
    text_mirror: bool = False,
    text_rotation: int = 0,
    text: ComponentSpec | None = text_rectangular,
    labels: tuple[str, ...] | None = None,
    **kwargs,
) -> Component:
    """Returns Component with 1D or 2D grid of components with text labels.

    Args:
        components: Iterable to be placed onto a grid. (can be 1D or 2D).
        text_prefix: for labels. For example. 'A' will produce 'A1', 'A2', ...
        text_offsets: relative to component anchor. Defaults to center.
        text_anchors: relative to component (ce cw nc ne nw sc se sw center cc).
        text_mirror: if True mirrors text.
        text_rotation: Optional text rotation.
        text: function to add text labels.
        labels: optional, specify a tuple of labels rather than using a text_prefix.

    keyword Args:
        spacing: between adjacent elements on the grid, can be a tuple for \
                different distances in height and width.
        separation: If True, guarantees elements are separated with fixed spacing \
                if False, elements are spaced evenly along a grid.
        shape: x, y shape of the grid (see np.reshape). \
                If no shape and the list is 1D, if np.reshape were run with (1, -1).
        align_x: {'x', 'xmin', 'xmax'} to perform the x (column) alignment along.
        align_y: {'y', 'ymin', 'ymax'} to perform the y (row) alignment along.
        edge_x: {'x', 'xmin', 'xmax'} to perform the x (column) distribution \
                ignored if separation = True.
        edge_y: {'y', 'ymin', 'ymax'} to perform the y (row) distribution along \
                ignored if separation = True.
        rotation: for each reference in degrees.


    .. plot::
        :include-source:

        import gdsfactory as gf

        components = [gf.components.triangle(x=i) for i in range(1, 10)]
        c = gf.grid_with_text(
            components,
            shape=(1, len(components)),
            rotation=0,
            h_mirror=False,
            v_mirror=True,
            spacing=(100, 100),
            text_offsets=((0, 100), (0, -100)),
            text_anchors=("nc", "sc"),
        )
        c.plot()

    """
    c = Component()
    g = grid(components=components, **kwargs)
    c << g
    if text:
        for i, ref in enumerate(g.named_references.values()):
            for text_offset, text_anchor in zip(text_offsets, text_anchors):
                if labels:
                    if len(labels) > i:
                        label = labels[i]
                    # skip labels for dummy components
                    else:
                        continue
                else:
                    label = f"{text_prefix}{i}"
                t = c << text(label)
                if text_mirror:
                    t.mirror()
                if text_rotation:
                    t.rotate(text_rotation)
                t.move(np.array(text_offset) + getattr(ref.size_info, text_anchor))
    return c


def test_grid() -> None:
    import gdsfactory as gf

    c = [gf.components.straight(length=i) for i in [1, 1, 2]]
    c = grid(
        c,
        shape=(2, 2),
        rotation=0,
        h_mirror=False,
        v_mirror=False,
        spacing=(10, 10),
    )
    # assert np.isclose(c.ports["1_1_o1"].center[0], 13.0), c.ports["1_1_o1"].center[0]
    assert c


if __name__ == "__main__":
    # test_grid()
    import gdsfactory as gf

    # components = [gf.components.rectangle(size=(i, i)) for i in range(40, 66, 5)]
    # components = [gf.components.rectangle(size=(i, i)) for i in range(40, 66, 5)]
    # c = [gf.components.triangle(x=i) for i in range(1, 10)]
    # print(len(c))

    c = grid([gf.components.straight(length=i) for i in range(1, 5)])
    # c = grid(
    #     c,
    #     shape=(2, 2),
    #     rotation=0,
    #     h_mirror=False,
    #     v_mirror=False,
    #     spacing=(10, 10),
    #     # text_offsets=((0, 100), (0, -100)),
    #     # text_anchors=("nc", "sc"),
    # )
    c.show(show_ports=True)
