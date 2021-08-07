"""wires for electrical manhattan routes
"""

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.hline import hline
from gdsfactory.cross_section import get_waveguide_settings
from gdsfactory.port import deco_rename_ports


@deco_rename_ports
@gf.cell
def wire_straight(
    length: float = 50.0,
    waveguide: gf.types.StrOrDict = "metal_routing",
    port_type: str = "dc",
    **kwargs
) -> Component:
    """Straight straight.

    Args:
        length: straiht length
        waveguide:
        port_type: port_type
        kwargs: waveguide_settings
    """
    waveguide_settings = get_waveguide_settings(waveguide=waveguide, **kwargs)
    width = waveguide_settings["width"]
    layer = waveguide_settings["layer"]

    c = hline(length=length, width=width, layer=layer, port_type=port_type)
    c.waveguide_settings = dict(layer=layer, width=width)
    return c


@deco_rename_ports
@gf.cell
def wire_corner(
    port_type: str = "dc", waveguide: gf.types.StrOrDict = "metal_routing", **kwargs
) -> Component:
    """90 degrees electrical corner

    Args:
        port_type: port_type
        waveguide:
        kwargs: waveguide_settings

    """
    waveguide_settings = get_waveguide_settings(waveguide=waveguide, **kwargs)
    width = waveguide_settings["width"]
    layer = waveguide_settings["layer"]

    c = Component()
    a = width / 2
    xpts = [-a, a, a, -a]
    ypts = [-a, -a, a, a]
    c.waveguide_settings = dict(layer=layer, width=width)

    c.add_polygon([xpts, ypts], layer=layer)
    c.add_port(
        name="W0",
        midpoint=(-a, 0),
        width=width,
        orientation=180,
        layer=layer,
        port_type=port_type,
    )
    c.add_port(
        name="N0",
        midpoint=(0, a),
        width=width,
        orientation=90,
        layer=layer,
        port_type=port_type,
    )
    c.length = width
    return c


if __name__ == "__main__":

    c = wire_straight()
    c = wire_corner()
    c.show(show_ports=True)
