import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_circular import grating_coupler_circular
from gdsfactory.types import ComponentOrFactory


@gf.cell
def grating_coupler_array(
    grating_coupler: ComponentOrFactory = grating_coupler_circular,
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "o1",
    rotation: int = 0,
) -> Component:
    """Array of rectangular pads.

    Args:
        grating_coupler: ComponentOrFactory
        spacing: x spacing
        n: number of pads
        port_name: port name
        rotation: rotation angle for each reference

    """
    c = Component()
    grating_coupler = (
        grating_coupler() if callable(grating_coupler) else grating_coupler
    )

    for i in range(n):
        gc = c << grating_coupler
        gc.rotate(rotation)
        gc.x = i * pitch
        port_name_new = f"o{i}"
        c.add_port(port=gc.ports[port_name], name=port_name_new)

    return c


if __name__ == "__main__":

    # c = gf.components.grating_coupler_circular()
    # c = gf.rotate(component=c, angle=90)
    # c = grating_coupler_array(grating_coupler=c, port_name="o1", pitch=25.0)
    c = grating_coupler_array()
    c.show(show_ports=True)
