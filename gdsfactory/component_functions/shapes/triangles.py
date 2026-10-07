from __future__ import annotations

__all__ = ["triangle", "triangle2", "triangle4"]

from typing import Any

from gdsfactory.component import Component
from gdsfactory.component_functions._get_component import get_component
from gdsfactory.typings import LayerSpec


def triangle(
    x: float = 10,
    xtop: float = 0,
    y: float = 20,
    ybot: float = 0,
    layer: LayerSpec = "WG",
) -> Component:
    r"""Return triangle.

    Args:
        x: base xsize.
        xtop: top xsize.
        y: ysize.
        ybot: bottom ysize.
        layer: layer.

    ```text
        xtop
           _
          | \
          |  \
          |   \
         y|    \
          |     \
          |      \
          |______|ybot
              x
    ```
    """
    c = Component()
    points = [(0, 0), (x, 0), (x, ybot), (xtop, y), (0, y)]
    c.add_polygon(points, layer=layer)
    return c


def triangle2(spacing: float = 3, **kwargs: Any) -> Component:
    r"""Return 2 triangles (bot, top).

    Args:
        spacing: between top and bottom.
        kwargs: triangle arguments.

    Keyword Args:
        x: base xsize.
        xtop: top xsize.
        y: ysize.
        ybot: bottom ysize.
        layer: layer.

          _
         | \
         |  \
         |   \
         |    \
         |     \
         |      \
         |       \
         |       |  spacing
         |      /
         |     /
         |    /
         |   /
         |  /
         |_/

    """
    c = Component()
    t = get_component("triangle", **kwargs)
    tt = c << t
    tb = c << t
    tb.dmirror()
    tb.rotate(180)
    tb.ymax = tt.ymin - spacing
    return c


def triangle4(**kwargs: Any) -> Component:
    r"""Return 4 triangles.

    Args:
        kwargs: triangle arguments.

    Keyword Args:
        x: base xsize.
        xtop: top xsize.
        y: ysize.
        ybot: bottom ysize.
        layer: layer.

                  / | \
                 /  |  \
                /   |   \
               /    |    \
              /     |     \
             /      |      \
            /       |       \
            |       |       |
            \       |      /
             \      |     /
              \     |    /
               \    |   /
                \   |  /
                 \  |_/

    """
    c = Component()
    t = get_component("triangle2", **kwargs)
    t1 = c << t
    t2 = c << t
    t2.dmirror()
    t2.xmax = t1.xmin
    return c
