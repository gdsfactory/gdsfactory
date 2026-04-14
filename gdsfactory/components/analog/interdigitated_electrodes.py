from __future__ import annotations

__all__ = ["interdigitated_electrodes"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags={"type": "analog"})
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
    c = Component()

    finger_pitch = finger_width + finger_gap
    total_finger_span = n_fingers * finger_width + (n_fingers - 1) * finger_gap

    if bus_length is None:
        bus_length = total_finger_span + 2 * finger_gap

    # Vertical extent:
    #   bottom bus: y in [-bus_width - finger_length - finger_gap/2, -finger_length - finger_gap/2]
    #   bottom fingers extend up from bottom bus
    #   top fingers extend down from top bus
    #   top bus: y in [finger_length + finger_gap/2, finger_length + finger_gap/2 + bus_width]

    gap_half = finger_gap / 2  # gap between finger tip and opposite bus
    top_bus_bottom = finger_length + gap_half
    top_bus_top = top_bus_bottom + bus_width
    bottom_bus_top = -(finger_length + gap_half)
    bottom_bus_bottom = bottom_bus_top - bus_width

    # Bottom bus bar
    c.add_polygon(
        [
            (-bus_length / 2, bottom_bus_bottom),
            (bus_length / 2, bottom_bus_bottom),
            (bus_length / 2, bottom_bus_top),
            (-bus_length / 2, bottom_bus_top),
        ],
        layer=layer,
    )

    # Top bus bar
    c.add_polygon(
        [
            (-bus_length / 2, top_bus_bottom),
            (bus_length / 2, top_bus_bottom),
            (bus_length / 2, top_bus_top),
            (-bus_length / 2, top_bus_top),
        ],
        layer=layer,
    )

    # Fingers: center the finger array horizontally
    x_start = -total_finger_span / 2 + finger_width / 2

    for i in range(n_fingers):
        x_center = x_start + i * finger_pitch
        x_left = x_center - finger_width / 2
        x_right = x_center + finger_width / 2

        if i % 2 == 0:
            # Bottom bus finger: extends upward from bottom bus top
            c.add_polygon(
                [
                    (x_left, bottom_bus_top),
                    (x_right, bottom_bus_top),
                    (x_right, bottom_bus_top + finger_length),
                    (x_left, bottom_bus_top + finger_length),
                ],
                layer=layer,
            )
        else:
            # Top bus finger: extends downward from top bus bottom
            c.add_polygon(
                [
                    (x_left, top_bus_bottom),
                    (x_right, top_bus_bottom),
                    (x_right, top_bus_bottom - finger_length),
                    (x_left, top_bus_bottom - finger_length),
                ],
                layer=layer,
            )

    # Electrical ports at bus bar ends
    c.add_port(
        name="bot_left",
        center=(-bus_length / 2, (bottom_bus_bottom + bottom_bus_top) / 2),
        width=bus_width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="bot_right",
        center=(bus_length / 2, (bottom_bus_bottom + bottom_bus_top) / 2),
        width=bus_width,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="top_left",
        center=(-bus_length / 2, (top_bus_bottom + top_bus_top) / 2),
        width=bus_width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="top_right",
        center=(bus_length / 2, (top_bus_bottom + top_bus_top) / 2),
        width=bus_width,
        orientation=0,
        layer=layer,
        port_type=port_type,
    )

    return c


if __name__ == "__main__":
    c = interdigitated_electrodes()
    c.show()
