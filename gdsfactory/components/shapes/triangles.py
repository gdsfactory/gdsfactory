from __future__ import annotations

__all__ = [
    "triangle",
    "triangle2",
    "triangle2_thin",
    "triangle4",
    "triangle4_thin",
    "triangle_thin",
]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["shapes"])
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
    return cf.triangle(x=x, xtop=xtop, y=y, ybot=ybot, layer=layer)


@gf.cell_with_module_name(tags=["shapes"])
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
    return cf.triangle2(spacing=spacing, **kwargs)


@gf.cell_with_module_name(tags=["shapes"])
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
    return cf.triangle4(**kwargs)


triangle_thin = CellAlias(triangle, xtop=0.2, x=2, y=5)
triangle2_thin = CellAlias(triangle2, xtop=0.2, x=2, y=5)
triangle4_thin = CellAlias(triangle4, xtop=0.2, x=2, y=5)
