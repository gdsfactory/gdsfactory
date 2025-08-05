from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpecs


@gf.cell_with_module_name
def copy_layers(
    factory: ComponentSpec = "cross",
    layers: LayerSpecs = ((1, 0), (2, 0)),
    flatten: bool = False,
    **kwargs: Any,
) -> Component:
    """Returns a component with the geometry copied in different layers.

    Args:
        factory: component spec.
        layers: iterable of layers.
        flatten: flatten the result.
        kwargs: keyword arguments passed to the component.
    """
    c = Component()

    ci = None
    for layer in layers:
        c << (ci := gf.get_component(factory, layer=layer, **kwargs))
    if ci is not None:
        c.copy_child_info(ci)

    if flatten:
        c.flatten()
    return c


if __name__ == "__main__":
    c = copy_layers(gf.components.rectangle)
    c.show()
