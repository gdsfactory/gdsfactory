from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.component_layout import _parse_layer
from gdsfactory.geometry.functions import polygon_grow
from gdsfactory.tech import LAYER
from gdsfactory.types import Layers


@cell
def add_keepout(
    component: Component,
    target_layers: Layers,
    keepout_layers: Layers,
    margin: float = 2.0,
) -> Component:
    """Adds keepout after looking up all polygons in a cell.

    You can also use add_padding.

    Args:
        component: to add keepout.
        target_layers: list of layers to read.
        keepout_layers: list of layers to add keepout.
        margin: offset from tareget to keepout_layers.
    """
    c = Component()
    c << component
    for layer in target_layers:
        polygons = component.get_polygons(by_spec=layer)
        if polygons:
            for ko_layer in keepout_layers:
                ko_layer = _parse_layer(ko_layer)
                polygon_keepout = [
                    polygon_grow(polygon, margin) for polygon in polygons
                ]
                c.add_polygon(polygon_keepout, ko_layer)
    return c


def test_add_keepout() -> None:
    from gdsfactory.components.straight import straight

    c = straight()
    polygons = len(c.get_polygons())
    target_layers = [LAYER.WG]
    keepout_layers = [LAYER.NO_TILE_SI]
    # print(len(c.get_polygons()))

    assert len(c.get_polygons()) == polygons
    c = add_keepout(
        component=c, target_layers=target_layers, keepout_layers=keepout_layers
    )
    # print(len(c.get_polygons()))
    assert len(c.get_polygons()) == polygons + 1


if __name__ == "__main__":
    # test_add_keepout()
    from gdsfactory.components.straight import straight

    c = straight()
    target_layers = [LAYER.WG]
    keepout_layers = [LAYER.SLAB150]
    c = add_keepout(c, target_layers, keepout_layers)
    c.show(show_ports=True)
