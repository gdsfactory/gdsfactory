from __future__ import annotations

__all__ = ["copy_layers"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, LayerSpecs


@gf.cell_with_module_name(tags=["containers"])
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
    return cf.copy_layers(
        factory=factory,
        layers=layers,
        flatten=flatten,
        **kwargs,
    )
