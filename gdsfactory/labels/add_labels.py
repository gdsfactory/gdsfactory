from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory.typings import LayerSpec, Ports


def add_port_labels(
    component: gf.Component,
    ports: Ports,
    layer: LayerSpec,
    texts: Sequence[str] | None = None,
) -> gf.Component:
    """Add port labels to a component.

    Args:
        component: to add the labels.
        ports: list of ports to add the labels.
        layer: layer to add the labels.
        texts: text to add to the labels. Defaults to port names.
    """
    texts = texts or [port.name for port in ports if port.name is not None]

    for text, port in zip(texts, ports, strict=False):
        component.add_label(
            text=text,
            position=port.center,
            layer=layer,
        )
    return component
