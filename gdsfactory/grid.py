"""pack a list of components into a grid.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import zip_longest
from typing import Literal

import kfactory as kf
import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import Anchor, ComponentSpec, ComponentSpecs, Float2, Spacing


def grid(
    components: ComponentSpecs = ("rectangle", "triangle"),
    spacing: Spacing | float = (5.0, 5.0),
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
                different distances in height and width or a single float.
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
            mirror=False,
            spacing=(100, 100),
        )
        c.plot()

    """
    c = gf.Component()
    instances = kf.grid(
        c,
        kcells=[gf.get_component(component) for component in components],
        shape=shape,
        spacing=(
            (float(spacing[0]), float(spacing[1]))
            if isinstance(spacing, tuple | list)
            else float(spacing)
        ),
        align_x=align_x,
        align_y=align_y,
        rotation=rotation,
        mirror=mirror,
    )
    for i, instance in enumerate(instances):
        c.add_ports(instance.ports, prefix=f"{i}_")
    return c


def grid_with_text(
    components: Sequence[ComponentSpec] = ("rectangle", "triangle"),
    text_prefix: str = "",
    text_offsets: Sequence[Float2] | None = None,
    text_anchors: Sequence[Anchor] | None = None,
    text_mirror: bool = False,
    text_rotation: int = 0,
    text: ComponentSpec | None = "text_rectangular",
    spacing: Spacing | float = (5.0, 5.0),
    shape: tuple[int, int] | None = None,
    align_x: Literal["origin", "xmin", "xmax", "center"] = "center",
    align_y: Literal["origin", "ymin", "ymax", "center"] = "center",
    rotation: int = 0,
    mirror: bool = False,
    labels: Sequence[str] | None = None,
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
        shape: x, y shape of the grid (see np.reshape).
        align_x: x alignment along (origin, xmin, xmax, center).
        align_y: y alignment along (origin, ymin, ymax, center).
        rotation: for each component in degrees.
        mirror: horizontal mirror y axis (x, 1) (1, 0). most common mirror.
        labels: list of labels for each component.


    .. plot::
        :include-source:

        import gdsfactory as gf

        components = [gf.components.triangle(x=i) for i in range(1, 10)]
        c = gf.grid_with_text(
            components,
            shape=(1, len(components)),
            rotation=0,
            mirror=False,
            spacing=(100, 100),
            text_offsets=((0, 100), (0, -100)),
            text_anchors=("nc", "sc"),
        )
        c.plot()

    """
    component_list = [gf.get_component(component) for component in components]
    text_offsets = text_offsets or ((0, 0),)
    text_anchors = text_anchors or ("center",)
    labels_not_none: list[str | None] = (
        list(labels) if labels else [None] * len(component_list)
    )

    if len(labels_not_none) != len(component_list):
        raise ValueError(
            f"Number of labels {len(labels_not_none)} must match number of components {len(component_list)}"
        )

    c = gf.Component()
    instances = kf.grid(
        c,
        kcells=component_list,
        shape=shape,
        spacing=(
            (float(spacing[0]), float(spacing[1]))
            if isinstance(spacing, tuple | list)
            else float(spacing)
        ),
        align_x=align_x,
        align_y=align_y,
        rotation=rotation,
        mirror=mirror,
    )
    for i, instance in enumerate(instances):
        c.add_ports(instance.ports, prefix=f"{i}_")
        text_string = labels_not_none[i] or f"{text_prefix}_{i}"

        if text:
            for text_offset, text_anchor in zip_longest(text_offsets, text_anchors):
                t = c << gf.get_component(text, text=text_string)
                size_info = instance.dsize_info
                text_offset = text_offset or (0, 0)
                text_anchor = text_anchor or "center"
                o = np.array(text_offset)
                d = np.array(getattr(size_info, text_anchor))
                t.move(tuple(o + d))
                if text_mirror:
                    t.dmirror()
                if text_rotation:
                    t.rotate(text_rotation)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # test_grid()
    # components = [gf.components.rectangle(size=(i, i)) for i in range(40, 66, 5)]
    # c = tuple(gf.components.rectangle(size=(i, i)) for i in range(40, 66, 10))
    components = tuple([gf.components.triangle(x=i) for i in range(1, 10)])
    # components = tuple(gf.components.rectangle(size=(i, i)) for i in range(1, 3))
    # print(len(c))

    c = grid(
        components,
        shape=(3, 3),
        rotation=90,
        mirror=False,
        spacing=(200.0, 200.0),
        # spacing=1,
        # text_offsets=((0, 100), (0, -100)),
        # labels=["r1", "r2"],
    )
    c.show()
