from __future__ import annotations

__all__ = ["cantilever"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["mems"])
def cantilever(
    beam_width: float = 2.0,
    beam_length: float = 20.0,
    anchor_width: float = 5.0,
    anchor_length: float = 5.0,
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """Returns a simple cantilever beam with an anchor.

    A rectangular anchor on the left with a thinner beam extending to the right.

    Args:
        beam_width: width of the cantilever beam.
        beam_length: length of the cantilever beam.
        anchor_width: width (vertical) of the anchor pad.
        anchor_length: length (horizontal) of the anchor pad.
        layer: layer spec.
        port_type: port type for electrical ports.
    """
    return cf.cantilever(
        beam_width=beam_width,
        beam_length=beam_length,
        anchor_width=anchor_width,
        anchor_length=anchor_length,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = cantilever()
    c.show()
