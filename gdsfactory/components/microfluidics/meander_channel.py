from __future__ import annotations

__all__ = ["meander_channel"]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def meander_channel(
    channel_width: float = 1.0,
    n_turns: int = 5,
    turn_spacing: float = 5.0,
    straight_length: float = 20.0,
    reservoir_length: float = 5.0,
    reservoir_height: float = 5.0,
    layer: LayerSpec = "WG",
    port_type: str | None = "optical",
) -> Component:
    """Returns a microfluidic meander (serpentine) channel.

    Builds a series of connected rectangles forming a serpentine path.
    Starts from the left, goes right for straight_length, turns up/down
    by turn_spacing, goes back left, and repeats for n_turns.
    Rectangular reservoirs are added at the inlet and outlet when
    reservoir_length > 0.

    Args:
        channel_width: width of the channel.
        n_turns: number of straight segments (horizontal passes).
        turn_spacing: center-to-center vertical spacing between passes.
        straight_length: length of each horizontal segment.
        reservoir_length: length of the reservoirs at inlet/outlet.
        reservoir_height: height of the reservoirs at inlet/outlet.
        layer: layer spec.
        port_type: None, optical, or electrical.
    """
    c = Component()
    hw = channel_width / 2

    x = 0.0
    y = 0.0

    for i in range(n_turns):
        going_right = i % 2 == 0

        # Horizontal segment
        if going_right:
            x_start = x
            x_end = x + straight_length
        else:
            x_start = x
            x_end = x - straight_length

        c.add_polygon(
            [
                (x_start, y - hw),
                (x_end, y - hw),
                (x_end, y + hw),
                (x_start, y + hw),
            ],
            layer=layer,
        )

        if going_right:
            x = x_end
        else:
            x = x_end

        # Vertical connector to next pass (skip after last segment)
        if i < n_turns - 1:
            y_next = y + turn_spacing
            cx = x
            c.add_polygon(
                [
                    (cx - hw, y - hw),
                    (cx + hw, y - hw),
                    (cx + hw, y_next + hw),
                    (cx - hw, y_next + hw),
                ],
                layer=layer,
            )
            y = y_next

    # Determine inlet and outlet positions
    # Inlet is at the start of the first segment
    inlet_x = 0.0
    inlet_y = 0.0

    # Outlet is at the end of the last segment
    last_going_right = (n_turns - 1) % 2 == 0
    outlet_x = x
    outlet_y = y

    # Add reservoirs
    if reservoir_length > 0:
        rh = reservoir_height / 2
        # Inlet reservoir (extends to the left)
        c.add_polygon(
            [
                (inlet_x - reservoir_length, inlet_y - rh),
                (inlet_x, inlet_y - rh),
                (inlet_x, inlet_y + rh),
                (inlet_x - reservoir_length, inlet_y + rh),
            ],
            layer=layer,
        )
        # Outlet reservoir (extends beyond the outlet)
        if last_going_right:
            c.add_polygon(
                [
                    (outlet_x, outlet_y - rh),
                    (outlet_x + reservoir_length, outlet_y - rh),
                    (outlet_x + reservoir_length, outlet_y + rh),
                    (outlet_x, outlet_y + rh),
                ],
                layer=layer,
            )
            outlet_port_x = outlet_x + reservoir_length
            outlet_orientation = 0
        else:
            c.add_polygon(
                [
                    (outlet_x - reservoir_length, outlet_y - rh),
                    (outlet_x, outlet_y - rh),
                    (outlet_x, outlet_y + rh),
                    (outlet_x - reservoir_length, outlet_y + rh),
                ],
                layer=layer,
            )
            outlet_port_x = outlet_x - reservoir_length
            outlet_orientation = 180

        inlet_port_x = inlet_x - reservoir_length
        inlet_port_width = reservoir_height
        outlet_port_width = reservoir_height
    else:
        inlet_port_x = inlet_x
        inlet_port_width = channel_width
        if last_going_right:
            outlet_port_x = outlet_x
            outlet_orientation = 0
        else:
            outlet_port_x = outlet_x
            outlet_orientation = 180
        outlet_port_width = channel_width

    if port_type:
        prefix = "o" if port_type == "optical" else "e"
        c.add_port(
            f"{prefix}1",
            center=(inlet_port_x, inlet_y),
            width=inlet_port_width,
            orientation=180,
            layer=layer,
            port_type=port_type,
        )
        c.add_port(
            f"{prefix}2",
            center=(outlet_port_x, outlet_y),
            width=outlet_port_width,
            orientation=outlet_orientation,
            layer=layer,
            port_type=port_type,
        )
        c.auto_rename_ports()

    return c


if __name__ == "__main__":
    c = meander_channel()
    c.show()
