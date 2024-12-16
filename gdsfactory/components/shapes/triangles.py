from __future__ import annotations

from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell
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

    .. code::

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
    """
    c = Component()
    points = [(0, 0), (x, 0), (x, ybot), (xtop, y), (0, y)]
    c.add_polygon(points, layer=layer)
    return c


@gf.cell
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

    .. code::

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
    t = triangle(**kwargs)
    tt = c << t
    tb = c << t
    tb.dmirror()
    tb.drotate(180)
    tb.dymax = tt.dymin - spacing
    return c


@gf.cell
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

    .. code::

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
    t = triangle2(**kwargs)
    t1 = c << t
    t2 = c << t
    t2.dmirror()
    t2.dxmax = t1.dxmin
    return c


triangle_thin = partial(triangle, xtop=0.2, x=2, y=5)
triangle2_thin = partial(triangle2, xtop=0.2, x=2, y=5)
triangle4_thin = partial(triangle2, xtop=0.2, x=2, y=5)

if __name__ == "__main__":
    # cc = triangle(xtop=5, ybot=5)
    cc = triangle4_thin(spacing=0)
    cc.show()
