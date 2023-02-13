from __future__ import annotations

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.cross import cross
from gdsfactory.typings import ComponentSpec, LayerSpecs


@cell
def copy_layers(
    factory: ComponentSpec = cross, layers: LayerSpecs = ((1, 0), (2, 0)), **kwargs
) -> Component:
    """Returns a component with the geometry copied in different layers.

    Args:
        factory: component spec.
        layers: iterable of layers.
        kwargs: keyword arguments.
    """
    c = Component()
    for layer in layers:
        ci = factory(layer=layer, **kwargs)
        c << ci

    c.copy_child_info(ci)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = copy_layers(gf.components.rectangle)
    c.show(show_ports=True)
