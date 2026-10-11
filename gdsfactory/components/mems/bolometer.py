from __future__ import annotations

__all__ = ["bolometer"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["mems"])
def bolometer(
    absorber_width: float = 20.0,
    absorber_length: float = 20.0,
    leg_width: float = 0.5,
    leg_length: float = 15.0,
    n_legs: int = 4,
    pad_width: float = 5.0,
    pad_length: float = 5.0,
    layer: LayerSpec = "WG",
    port_type: str = "electrical",
) -> Component:
    """Returns a bolometer thermal detector.

    A central absorber rectangle with thin L-shaped support legs extending
    outward to anchor pads distributed around the perimeter.

    Args:
        absorber_width: width of the central absorber.
        absorber_length: length of the central absorber.
        leg_width: width of each support leg.
        leg_length: length of each support leg.
        n_legs: number of support legs (distributed around perimeter).
        pad_width: width of each anchor pad.
        pad_length: length of each anchor pad.
        layer: layer spec.
        port_type: port type for electrical ports.
    """
    return cf.bolometer(
        absorber_width=absorber_width,
        absorber_length=absorber_length,
        leg_width=leg_width,
        leg_length=leg_length,
        n_legs=n_legs,
        pad_width=pad_width,
        pad_length=pad_length,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = bolometer()
    c.show()
