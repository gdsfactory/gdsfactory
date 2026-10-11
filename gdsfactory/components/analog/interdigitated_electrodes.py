from __future__ import annotations

__all__ = ["interdigitated_electrodes"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["analog"])
def interdigitated_electrodes(
    n_fingers: int = 10,
    finger_width: float = 0.5,
    finger_length: float = 10.0,
    finger_gap: float = 0.5,
    bus_width: float = 2.0,
    bus_length: float | None = None,
    layer: LayerSpec = "MTOP",
    port_type: str = "electrical",
) -> Component:
    """Interdigitated electrode pattern.

    Two horizontal bus bars (top and bottom) with alternating fingers extending
    from each bus toward the opposite one. Fingers from the top bus extend
    downward and fingers from the bottom bus extend upward, interleaving
    with a gap between the finger tips and the opposite bus.

    Args:
        n_fingers: Total number of fingers (split between top and bottom buses).
        finger_width: Width of each finger in um.
        finger_length: Length of each finger in um.
        finger_gap: Gap between adjacent fingers (edge to edge) in um.
        bus_width: Width (height) of each bus bar in um.
        bus_length: Length of each bus bar in um. Defaults to the total width
            needed to accommodate all fingers.
        layer: Layer specification for all geometry.
        port_type: Port type for electrical ports at bus bar ends.
    """
    return cf.interdigitated_electrodes(
        n_fingers=n_fingers,
        finger_width=finger_width,
        finger_length=finger_length,
        finger_gap=finger_gap,
        bus_width=bus_width,
        bus_length=bus_length,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = interdigitated_electrodes()
    c.show()
