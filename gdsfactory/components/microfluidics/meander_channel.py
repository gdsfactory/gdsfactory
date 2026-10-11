from __future__ import annotations

__all__ = ["meander_channel"]

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["microfluidics"])
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
    return cf.meander_channel(
        channel_width=channel_width,
        n_turns=n_turns,
        turn_spacing=turn_spacing,
        straight_length=straight_length,
        reservoir_length=reservoir_length,
        reservoir_height=reservoir_height,
        layer=layer,
        port_type=port_type,
    )


if __name__ == "__main__":
    c = meander_channel()
    c.show()
