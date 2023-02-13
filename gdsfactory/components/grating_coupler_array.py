from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical import grating_coupler_elliptical
from gdsfactory.typings import ComponentSpec


@gf.cell
def grating_coupler_array(
    grating_coupler: ComponentSpec = grating_coupler_elliptical,
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
    rotation: int = 0,
) -> Component:
    """Array of rectangular pads.

    Args:
        grating_coupler: ComponentSpec.
        pitch: x spacing.
        n: number of pads.
        port_name: port name.
        rotation: rotation angle for each reference.
    """
    c = Component()
    grating_coupler = gf.get_component(grating_coupler)

    for i in range(n):
        gc = c << grating_coupler
        gc.rotate(rotation)
        gc.x = i * pitch
        port_name_new = f"o{i}"
        c.add_port(port=gc.ports[port_name], name=port_name_new)

    return c


if __name__ == "__main__":
    c = grating_coupler_array()
    c.show(show_ports=True)
