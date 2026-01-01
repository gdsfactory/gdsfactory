"""You can remove a list of layers from a component."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.gpdk import PDK

PDK.activate()


@gf.cell
def remove_layers_overlapping() -> Component:
    c = gf.Component()

    _ = c << gf.components.via_stack_heater_m3()
    ref2 = c << gf.components.via_stack_npp_m1()
    ref2.move((5, 0))

    c = gf.functions.remove_shapes_near_exclusion(
        c,
        target_layer="VIA2",
        exclusion_layer="VIAC",
        margin=1,
    )
    return c


if __name__ == "__main__":
    c = remove_layers_overlapping()
    c.show()
