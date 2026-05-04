"""This module contains the fixed cells from the AMF library."""

import gdsfactory as gf
from gdsfactory.typings import (
    Size,
)


@gf.cell
def via_stack_heater_mtop(size: Size = (20.0, 10.0)) -> gf.Component:
    """Rectangular via array stack.

    You can use it to connect different metal layers or metals to silicon.
    You can use the naming convention via_stack_layerSource_layerDestination
    contains 4 ports (e1, e2, e3, e4)

    also know as Via array
    http://www.vlsi-expert.com/2017/12/vias.html

    Args:
        size: of the layers.
    """
    return gf.c.via_stack(
        size=size,
        layers=("HEATER", "PAD"),
        layer_offsets=None,
        vias=(None, None),
        layer_to_port_orientations=None,
        correct_size=True,
        slot_horizontal=False,
        slot_vertical=False,
        port_orientations=(180, 90, 0, -90),
    )
