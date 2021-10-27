import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.triangle import triangle
from gdsfactory.tech import LAYER
from gdsfactory.types import ComponentFactory, Float2, Layer

triangle_metal = gf.partial(triangle, layer=LAYER.M3, xtop=2)


@gf.cell
def dicing_lane(
    size: Float2 = (50, 300),
    marker: ComponentFactory = triangle_metal,
    layer_dicing: Layer = (100, 0),
) -> Component:
    """Dicing lane with triangular markers on both sides.

    Args:
        size: (tuple) Width and height of rectangle.
        marker: function to generate the dicing lane markers
        layer_dicing: Specific layer to put polygon geometry on.

    """
    c = Component()
    r = c << rectangle(size=size, layer=layer_dicing)

    m = marker()

    mbr = c << m
    mbr.xmin = r.xmax

    mbl = c << m
    mbl.mirror()
    mbl.xmax = r.xmin

    mtr = c << m
    mtr.mirror()
    mtr.rotate(180)
    mtr.xmin = r.xmax
    mtr.ymax = r.ymax

    mtl = c << m
    mtl.rotate(180)
    mtl.xmax = r.xmin
    mtl.ymax = r.ymax
    return c


if __name__ == "__main__":
    c = dicing_lane()
    c.show()
