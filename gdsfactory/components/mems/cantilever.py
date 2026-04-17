from __future__ import annotations

__all__ = ["cantilever"]

import gdsfactory as gf
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
    c = Component()

    # Anchor rectangle on the left, centered vertically at y=0
    c.add_polygon(
        [
            (0, -anchor_width / 2),
            (anchor_length, -anchor_width / 2),
            (anchor_length, anchor_width / 2),
            (0, anchor_width / 2),
        ],
        layer=layer,
    )

    # Beam rectangle extending right from anchor, centered vertically at y=0
    c.add_polygon(
        [
            (anchor_length, -beam_width / 2),
            (anchor_length + beam_length, -beam_width / 2),
            (anchor_length + beam_length, beam_width / 2),
            (anchor_length, beam_width / 2),
        ],
        layer=layer,
    )

    # Port at the left edge of the anchor
    c.add_port(
        "e1",
        center=(0, 0),
        width=anchor_width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )

    # Port at the free tip of the beam
    c.add_port(
        "e2",
        center=(anchor_length + beam_length, 0),
        width=beam_width,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )

    return c


if __name__ == "__main__":
    c = cantilever()
    c.show()
