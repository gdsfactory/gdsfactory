from __future__ import annotations

__all__ = ["anchored_flexure"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["mems"])
def anchored_flexure(
    hinge_width: float = 0.3,
    hinge_length: float = 5.0,
    pad_width: float = 10.0,
    pad_length: float = 10.0,
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """Returns a flexure hinge between two pads.

    Two rectangular pads connected by a thin hinge, all centered vertically.

    Args:
        hinge_width: width of the thin flexure hinge.
        hinge_length: length of the hinge connecting the two pads.
        pad_width: width (vertical) of each pad.
        pad_length: length (horizontal) of each pad.
        layer: layer spec.
        port_type: port type for electrical ports.
    """
    return cf.anchored_flexure(
        hinge_width=hinge_width,
        hinge_length=hinge_length,
        pad_width=pad_width,
        pad_length=pad_length,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = anchored_flexure()
    c.show()
