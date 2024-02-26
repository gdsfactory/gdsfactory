"""Via chain."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_pins import LayerSpec
from gdsfactory.component import Component
from gdsfactory.components.via import via1
from gdsfactory.components.via_stack import via_stack_m2_m3
from gdsfactory.typings import ComponentSpec


@gf.cell
def via_chain(
    num_vias: float = 100.0,
    cols: int = 10,
    wire_width: float = 2.0,
    via: ComponentSpec = via1,
    contact: ComponentSpec = via_stack_m2_m3,
    layer_bot: LayerSpec = "M1",
    layer_top: LayerSpec = "M2",
    via_min_enclosure: float = 1.0,
    min_metal_spacing: float = 1.0,
) -> Component:
    """Via chain to extract via resistance.

    Args:
        num_vias: number of vias.
        cols_pairs: number of column pairs.
        wire_width: width of wire.
        via: via component.
        contact: contact component.
        layer_bot: bottom layer.
        layer_top: top layer.
        via_min_enclosure: via_min_enclosure.
        min_metal_spacing: min_metal_spacing.

    .. code::

        side view:

           ┌────────────────────────────────────┐  spacing     ┌────────────────────────────────────┐
           │                                    │              │                                    │
           │                                    │◄───────────► │                                    │
           └─────────────┬─────┬────────────────┘              └───────────────┬─────┬──────────────┘
                         │     │         via_enclosure                         │     │
                         │     │◄───────────────►                              │     │
                         │     │                                               │     │
                         │     │                                               │     │
                         │width│                                               │     │
                         ◄─────►                                               │     │
                         │     │                                               │     │
           ┌─────────────┴─────┴───────────────────────────────────────────────┴─────┴───────────────┐
           │                                                                                         │
           │                                                                                         │
           └─────────────────────────────────────────────────────────────────────────────────────────┘

           ◄─────────────────────────────────────────────────────────────────────────────────────────►
                                         2*e + w + spacing + 2*e + w

    """

    if cols % 2 != 0:
        raise ValueError(f"{cols=} must be even")

    c = gf.Component()
    rows = num_vias / cols

    if int(rows) != rows:
        raise ValueError(f"{num_vias=} must be a multiple of {cols=}")

    if rows <= 1:
        raise ValueError(
            f"rows must be at least 2. Got {rows=}. You can increase the number vias {num_vias=}."
        )

    if rows % 2 != 0:
        raise ValueError(
            f"{rows=} must be even. Number of vias needs to be a multiple of {2*cols=}."
        )

    via = gf.get_component(via)
    contact = gf.get_component(contact)
    wire_length = 2 * (2 * via_min_enclosure + via.size_info.width) + min_metal_spacing
    via_width = via.size_info.width

    wire_size = (wire_length, wire_width)
    via_spacing = (
        2 * via_min_enclosure + min_metal_spacing + via.size_info.width,
        wire_width + min_metal_spacing,
    )

    vias = c.add_array(
        component=via,
        columns=cols,
        rows=rows,
        spacing=via_spacing,
    )
    top_wire = gf.c.rectangle(size=wire_size, layer=layer_top)
    top_wires = c.add_array(
        component=top_wire,
        columns=cols // 2,
        rows=rows,
        spacing=(wire_length + min_metal_spacing, wire_width + min_metal_spacing),
    )
    bot_wire = gf.c.rectangle(size=wire_size, layer=layer_bot)
    bot_wires = c.add_array(
        component=bot_wire,
        columns=cols // 2,
        rows=rows,
        spacing=(wire_length + min_metal_spacing, wire_width + min_metal_spacing),
    )
    top_wires.xmin = -via_min_enclosure
    bot_wires.xmin = top_wires.xmin + wire_length / 2 + min_metal_spacing / 2
    bot_wires.ymin = -via_min_enclosure
    top_wires.ymin = -via_min_enclosure
    vias.xmin = top_wires.xmin + via_min_enclosure + via_spacing[0]

    vertical_wire_left = gf.c.rectangle(
        size=(2 * via_min_enclosure + via_width, 2 * wire_width + min_metal_spacing),
        layer=layer_top,
    )

    right_wires = c.add_array(
        component=vertical_wire_left,
        columns=1,
        rows=rows // 2,
        spacing=(wire_width + min_metal_spacing, 2 * (wire_width + min_metal_spacing)),
    )

    right_wires.xmax = bot_wires.xmax
    right_wires.ymin = bot_wires.ymin

    left_wires = c.add_array(
        component=vertical_wire_left,
        columns=1,
        rows=rows // 2 - 1,
        spacing=(wire_width + min_metal_spacing, 2 * (wire_width + min_metal_spacing)),
    )

    left_wires.xmin = top_wires.xmin
    left_wires.ymin = bot_wires.ymin + wire_width + min_metal_spacing

    contact1 = c << contact
    contact2 = c << contact

    contact1.xmax = top_wires.xmin
    contact2.xmax = top_wires.xmin

    contact1.ymax = top_wires.ymin + wire_width
    contact2.ymin = top_wires.ymax - wire_width

    return c


if __name__ == "__main__":
    c = via_chain(num_vias=40)
    c.show(show_ports=True)
