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
    m = gf.get_component(marker)
    layers = layers or [layer_dicing]

    for layer in layers:
        r = c << rectangle(size=size, layer=layer, port_type=None)

        mbr = c << m
        mbr.dxmin = r.dxmax

        mbl = c << m
        mbl.dmirror()
        mbl.dxmax = r.dxmin
        mbl.dymin = r.dymin

        mtr = c << m
        mtr.dmirror_y()
        mtr.dxmin = r.dxmax
        mtr.dymax = r.dymax

        mtl = c << m
        mtl.drotate(180)
        mtl.dxmax = r.dxmin
        mtl.dymax = r.dymax
    c.flatten()
    return c


if __name__ == "__main__":
    c = dicing_lane()
    c.show()
