import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical2 import (
    grating_coupler_elliptical2,
)
from gdsfactory.types import ComponentOrFactory


@gf.cell
def grating_coupler_array(
    grating_coupler: ComponentOrFactory = grating_coupler_elliptical2,
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
) -> Component:
    """Array of rectangular pads.

    Args:
        grating_coupler: ComponentOrFactory
        spacing: x spacing
        n: number of pads
        port_list: list of port orientations (N, S, W, E) per pad

    """
    c = Component()
    grating_coupler = (
        grating_coupler() if callable(grating_coupler) else grating_coupler
    )

    for i in range(n):
        gc = c << grating_coupler
        gc.x = i * pitch
        port_name_new = f"o{i}"
        c.add_port(port=gc.ports[port_name], name=port_name_new)

    return c


if __name__ == "__main__":

    # c = gf.components.grating_coupler_elliptical2()
    # c = gf.rotate(component=c, angle=90)
    # c = grating_coupler_array(grating_coupler=c, port_name="o1", pitch=25.0)
    c = grating_coupler_array()
    c.show(show_ports=True)
