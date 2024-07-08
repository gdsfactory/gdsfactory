"""pack a list of components into a grid.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

from typing import Literal

import kfactory as kf
import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.text_rectangular import text_rectangular
from gdsfactory.components.triangles import triangle
from gdsfactory.typings import Anchor, ComponentSpec, ComponentSpecs, Float2


def grid(
    components: ComponentSpecs = (rectangle, triangle),
    spacing: tuple[float, float] | float = (5.0, 5.0),
    shape: tuple[int, int] | None = None,
    align_x: Literal["origin", "xmin", "xmax", "center"] = "center",
    align_y: Literal["origin", "ymin", "ymax", "center"] = "center",
    rotation: int = 0,
    mirror: bool = False,
) -> Component:
    """Returns Component with a 1D or 2D grid of components.

    Args:
        components: Iterable to be placed onto a grid. (can be 1D or 2D).
        spacing: between adjacent elements on the grid, can be a tuple for \
                different distances in height and width.
        shape: x, y shape of the grid (see np.reshape). \
                If no shape and the list is 1D, if np.reshape were run with (1, -1).
        align_x: x alignment along (origin, xmin, xmax, center).
        align_y: y alignment along (origin, ymin, ymax, center).
        rotation: for each component in degrees.
        mirror: horizontal mirror y axis (x, 1) (1, 0). most common mirror.

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
    c = gf.Component()
    instances = kf.grid(
        c,
        kcells=[gf.get_component(component) for component in components],
        shape=shape,
        spacing=(float(spacing[0]), float(spacing[1]))
        if isinstance(spacing, tuple | list)
        else float(spacing),
        align_x=align_x,
        align_y=align_y,
        rotation=round(rotation // 90),
        mirror=mirror,
    )
    for i, instances_list in enumerate(instances):
        for j, instance in enumerate(instances_list):
            # print(i, j)
            # instance.ports.print()
            c.add_ports(instance.ports, prefix=f"{j}_{i}_")
    return c


def grid_with_text(
    components: tuple[ComponentSpec, ...] = (rectangle, triangle),
    text_prefix: str = "",
    text_offsets: tuple[Float2, ...] | None = None,
    text_anchors: tuple[Anchor, ...] | None = None,
    text_mirror: bool = False,
    text_rotation: int = 0,
    text: ComponentSpec | None = text_rectangular,
    spacing: tuple[float, float] | float = (5.0, 5.0),
    shape: tuple[int, int] | None = None,
    align_x: Literal["origin", "xmin", "xmax", "center"] = "center",
    align_y: Literal["origin", "ymin", "ymax", "center"] = "center",
    rotation: int = 0,
    mirror: bool = False,
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
        spacing: between adjacent elements on the grid, can be a tuple for \
                different distances in height and width.
        shape: x, y shape of the grid (see np.reshape). \
        align_x: x alignment along (origin, xmin, xmax, center).
        align_y: y alignment along (origin, ymin, ymax, center).
        rotation: for each component in degrees.
        mirror: horizontal mirror y axis (x, 1) (1, 0). most common mirror.


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
    components = [gf.get_component(component) for component in components]
    text_offsets = text_offsets or [(0, 0)] * len(components)
    text_anchors = text_anchors or ["center"] * len(components)
    c = gf.Component()
    instances = kf.grid(
        c,
        kcells=components,
        shape=shape,
        spacing=(float(spacing[0]), float(spacing[1]))
        if isinstance(spacing, tuple | list)
        else float(spacing),
        align_x=align_x,
        align_y=align_y,
        rotation=round(rotation // 90),
        mirror=mirror,
    )
    for i, instances_list in enumerate(instances):
        for j, instance in enumerate(instances_list):
            c.add_ports(instance.ports, prefix=f"{j}_{i}_")
            if text:
                t = c << text(f"{text_prefix}{j}_{i}")
                size_info = instance.dsize_info
                o = np.array(text_offsets[j])
                d = np.array(getattr(size_info, text_anchors[j]))
                t.dmove(o + d)
                if text_mirror:
                    t.dmirror()
                if text_rotation:
                    t.drotate(text_rotation)

    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # test_grid()
    # components = [gf.components.rectangle(size=(i, i)) for i in range(40, 66, 5)]
    # c = tuple(gf.components.rectangle(size=(i, i)) for i in range(40, 66, 10))
    # c = tuple([gf.components.triangle(x=i) for i in range(1, 10)])
    c = tuple(gf.components.rectangle(size=(i, i)) for i in range(1, 10))
    # print(len(c))

    c = grid_with_text(
        c,
        shape=(3, 3),
        # rotation=90,
        mirror=False,
        spacing=(200.0, 200.0),
        # spacing=1,
        # text_offsets=((0, 100), (0, -100)),
    )
    c.show()
