"""Based on phidl.geometry."""

from __future__ import annotations

import gdstk

import gdsfactory as gf
from gdsfactory.component_layout import Polygon
from gdsfactory.typings import Component, ComponentReference, LayerSpec


@gf.cell
def offset(
    elements: Component,
    distance: float = 0.1,
    use_union: bool = True,
    precision: float = 1e-4,
    join: str = "miter",
    tolerance: int = 2,
    layer: LayerSpec = "WG",
) -> Component:
    """Returns new Component with polygons eroded or dilated by an offset.

    Args:
        elements: Component(/Reference), list of Component(/Reference), or Polygon
          Polygons to offset or Component containing polygons to offset.
        distance: Distance to offset polygons. Positive values expand, negative shrink.
        precision: Desired precision for rounding vertex coordinates.
        join: {'miter', 'bevel', 'round'} Type of join used to create polygon offset
        tolerance: For miter joints, this number must be at least 2 represents the
          maximal distance in multiples of offset between new vertices and their
          original position before beveling to avoid spikes at acute joints. For
          round joints, it indicates the curvature resolution in number of
          points per full circle.
        layer: Specific layer to put polygon geometry on.

    Returns:
        Component containing a polygon(s) with the specified offset applied.

    .. plot::
      :include-source:

      import gdsfactory as gf
      c = gf.Component()
      layer_slab = (2, 0)
      c1 = gf.components.coupler_ring(
          cladding_layers=[layer_slab], cladding_offsets=[0.5]
      )
      d = 0.8
      c2 = gf.geometry.offset(c1, distance=+d, layer=layer_slab)
      c3 = gf.geometry.offset(c2, distance=-d, layer=layer_slab)

      c << c1.extract(layers=("WG",))
      c << c3
      c.plot_matplotlib()

    """
    if not isinstance(elements, list):
        elements = [elements]
    polygons_to_offset = []
    for e in elements:
        if isinstance(e, (Component, ComponentReference)):
            polygons_to_offset += e.get_polygons(by_spec=False)
        elif isinstance(e, (Polygon, gdstk.Polygon)):
            polygons_to_offset.append(e)
    if len(polygons_to_offset) == 0:
        return gf.Component("offset")

    gds_layer, gds_datatype = gf.get_layer(layer)
    p = gdstk.offset(
        polygons_to_offset,
        distance=distance,
        join=join,
        tolerance=tolerance,
        precision=precision,
        use_union=use_union,
        layer=gds_layer,
        datatype=gds_datatype,
    )

    component = gf.Component()
    component.add_polygon(p, layer=layer)
    return component


def test_offset() -> None:
    c = gf.components.ring()
    co = offset(c, distance=0.5)
    assert int(co.area()) == 94


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    layer_slab = (2, 0)
    c1 = gf.components.coupler_ring(
        cladding_layers=[layer_slab], cladding_offsets=[0.5]
    )
    d = 0.8
    # d = 1
    c2 = gf.geometry.offset(c1, distance=+d, layer=layer_slab)
    c3 = gf.geometry.offset(c2, distance=-d, layer=layer_slab)

    c << c1.extract(layers=("WG",))
    c << c3
    c.show()
