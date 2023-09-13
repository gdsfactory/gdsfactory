"""Based on phidl.geometry."""
from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.geometry.boolean import boolean
from gdsfactory.typings import LayerSpec


@gf.cell
def invert(
    elements,
    border: float = 10.0,
    precision: float = 1e-4,
    layer: LayerSpec = (1, 0),
) -> Component:
    """Returns inverted version of input shapes with additional border around the edges.

    Args:
        elements : Component(/Reference), list of Component(/Reference), or Polygon \
                A Component containing the polygons to invert.
        border: Size of the border around the inverted shape (border value is the \
                distance from the edges of the boundary box defining the inverted \
                shape to the border, and is applied to all 4 sides of the shape).
        precision: Desired precision for rounding vertex coordinates.
        layer: Specific layer(s) to put polygon geometry on.

    Returns
        Component with inverted version of the input shape(s) and the border(s).

    .. plot::
      :include-source:

      import gdsfactory as gf

      e1 = gf.components.ellipse(radii=(6, 6))
      c = gf.geometry.invert(e1)
      c.plot_matplotlib()

    """
    Temp = Component()
    if type(elements) is not list:
        elements = [elements]
    for e in elements:
        if isinstance(e, Component):
            Temp.add_ref(e)
        else:
            Temp.add(e)
    gds_layer, gds_datatype = gf.get_layer(layer)

    # Build the rectangle around the Component D
    R = gf.components.rectangle(
        size=(Temp.xsize + 2 * border, Temp.ysize + 2 * border), centered=True
    ).ref()
    R.center = Temp.center
    return boolean(
        A=R,
        B=Temp,
        operation="A-B",
        precision=precision,
        layer=layer,
    )


def test_invert() -> None:
    e1 = gf.components.ellipse(radii=(6, 6))
    c = invert(e1)
    assert int(c.area()) == 910


if __name__ == "__main__":
    # test_invert()
    e1 = gf.components.ellipse(radii=(6, 6))
    c = invert(e1)
    c.show(show_ports=True)
