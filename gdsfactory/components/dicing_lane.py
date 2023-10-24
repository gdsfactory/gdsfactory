from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.triangles import triangle
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec, LayerSpecs

triangle_metal = partial(triangle, layer="MTOP", xtop=2)


@gf.cell
def dicing_lane(
    size: Float2 = (50, 300),
    marker: ComponentSpec = triangle_metal,
    layer_dicing: LayerSpec = "DICING",
    layers: LayerSpecs | None = None,
) -> Component:
    """Dicing lane with triangular markers on both sides.

    Args:
        size: (tuple) Width and height of rectangle.
        marker: function to generate the dicing lane markers.
        layer_dicing: Specific layer to put polygon geometry on.
        layers: optional list of layers to duplicate the geometry.
    """
    c = Component()

    layers = layers or [layer_dicing]

    for layer in layers:
        m = gf.get_component(marker, layer=layer)
        r = c << rectangle(size=size, layer=layer)

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
    layers = {(1, 0), (2, 0)}
    c = dicing_lane()
    c.show(show_ports=True)
