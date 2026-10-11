from __future__ import annotations

__all__ = ["copy_layers"]

from typing import Any

from gdsfactory.component import Component
from gdsfactory.component_functions._get_component import get_component
from gdsfactory.typings import ComponentSpec, LayerSpecs


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
        c << (ci := get_component(factory, layer=layer, **kwargs))
    if ci is not None:
        c.copy_child_info(ci)

    if flatten:
        c.flatten()
    return c
