from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpecs


@gf.cell
def copy_layers(
    factory: ComponentSpec = "cross",
    layers: LayerSpecs = ((1, 0), (2, 0)),
    **kwargs: Any,
) -> Component:
    """Returns a component with the geometry copied in different layers.

    Args:
        factory: component spec.
        layers: iterable of layers.
        kwargs: keyword arguments.
    """
    c = Component()
    for layer in layers:
        ci = gf.get_component(factory, layer=layer, **kwargs)
        _ = c << ci

    c.copy_child_info(ci)
    return c


if __name__ == "__main__":
    c = copy_layers(gf.components.rectangle)
    c.show()
