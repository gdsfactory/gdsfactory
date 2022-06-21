from functools import partial

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.types import LayerSpec


@cell
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
    points = [[0, 0], [x, 0], [x, ybot], [xtop, y], [0, y]]
    c.add_polygon(points, layer=layer)
    return c


@cell
def triangle2(spacing: float = 3, **kwargs):
    r"""Return 2 triangles (bot, top).

    Args:
        spacing: between top and bottom.

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
    tb.mirror()
    tb.rotate(180)
    tb.ymax = tt.ymin - spacing
    return c


@cell
def triangle4(**kwargs):
    r"""Return 4 triangles.

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
    t2.mirror()
    t2.xmax = t1.xmin
    return c


triangle_thin = partial(triangle, xtop=0.2, x=2, y=5)
triangle2_thin = partial(triangle2, xtop=0.2, x=2, y=5)
triangle4_thin = partial(triangle2, xtop=0.2, x=2, y=5)

if __name__ == "__main__":
    # cc = triangle(xtop=5, ybot=5)
    cc = triangle4_thin(spacing=0)
    cc.show(show_ports=True)
