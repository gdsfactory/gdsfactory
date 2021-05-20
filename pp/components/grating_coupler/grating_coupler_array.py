import pp
from pp.component import Component
from pp.components.grating_coupler.elliptical2 import grating_coupler_elliptical2
from pp.types import ComponentOrFactory


@pp.cell_with_validator
def grating_coupler_array(
    grating_coupler: ComponentOrFactory = grating_coupler_elliptical2,
    pitch: float = 127.0,
    n: int = 6,
    port_name: str = "W0",
    **grating_coupler_settings,
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
        grating_coupler(**grating_coupler_settings)
        if callable(grating_coupler)
        else grating_coupler
    )

    for i in range(n):
        gc = c << grating_coupler
        gc.x = i * pitch
        port_name_new = f"{port_name}{i}"
        c.add_port(port=gc.ports[port_name], name=port_name_new)

    return c


if __name__ == "__main__":
    import pp

    c = pp.c.grating_coupler_elliptical2()
    c = pp.rotate(component=c, angle=90)
    c = grating_coupler_array(grating_coupler=c, port_name="S0", pitch=25.0)
    c.show(show_ports=True)
