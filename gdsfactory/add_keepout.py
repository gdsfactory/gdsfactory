from __future__ import annotations

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.component_layout import _parse_layer
from gdsfactory.generic_tech import LAYER
from gdsfactory.geometry.functions import polygon_grow
from gdsfactory.typings import Layers


@cell
def add_keepout(
    component: Component,
    target_layers: Layers,
    keepout_layers: Layers,
    margin: float = 2.0,
) -> Component:
    """Adds keepout after looking up all polygons in a cell.

    You can also use add_padding for rectangular keepout.

    Args:
        component: to add keepout.
        target_layers: list of layers to read.
        keepout_layers: list of layers to add keepout.
        margin: offset from target to keepout_layers.
    """
    c = Component()
    c << component
    for layer in target_layers:
        polygons = component.get_polygons(by_spec=layer, as_array=False)
        if polygons:
            for ko_layer in keepout_layers:
                ko_layer = _parse_layer(ko_layer)

                for polygon in polygons:
                    polygon_keepout = polygon_grow(polygon.points, margin)
                    c.add_polygon(points=polygon_keepout, layer=ko_layer)

    return c


def test_add_keepout() -> None:
    from gdsfactory.components.straight import straight

    c = straight()
    polygons = len(c.get_polygons())
    target_layers = [LAYER.WG]
    keepout_layers = [LAYER.NO_TILE_SI]

    assert len(c.get_polygons()) == polygons
    assert add_keepout(
        component=c, target_layers=target_layers, keepout_layers=keepout_layers
    )


if __name__ == "__main__":
    from gdsfactory.components.straight import straight

    c = straight()
    target_layers = [LAYER.WG]
    keepout_layers = [LAYER.SLAB150]
    c = add_keepout(c, target_layers, keepout_layers)
    c.show(show_ports=True)
