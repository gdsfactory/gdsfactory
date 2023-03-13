from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import Component, LayerSpecs, LayerSpec, Union


@gf.cell
def remove_layers(
    elements: Component,
    layers: Union[LayerSpec, LayerSpecs] = "WG",
) -> Component:
    """Returns a component with the same layers as the original element
    but with the layers in the 'layers' list removed.

    Args:
        elements: Component(/Reference), list of Component(/Reference), or Polygon
          Polygons to remove the layers from.
        layers: List of layers to remove

    Returns:
        Component containing a polygon(s) with the specified layers removed.

    """

    if type(layers) is not list:
        layers = [layers]

    layers = [gf.get_layer(layer) for layer in layers]

    if type(elements) is not list:
        elements = [elements]

    new_elems = list()

    for c in elements:
        c = c.extract(
            [
                gf.get_layer(layer)
                for layer in c.get_layers()
                if gf.get_layer(layer) not in layers
            ]
        )
        new_elems.append(c)

    if len(new_elems) == 1:
        new_elems = new_elems[0]

    return new_elems


def test_remove() -> None:
    c = gf.components.straight(cross_section="rib")
    c = remove_layers(c, "SLAB90")


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    c << gf.components.straight(cross_section="rib")

    c = remove_layers(c, "WG")

    c.show()
