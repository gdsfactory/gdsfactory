from typing import Optional, Tuple

import picwriter.components as pc

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.waveguide_template import strip
from gdsfactory.types import ComponentFactory, Layer


@gf.cell
def spiral(
    port_spacing: float = 500.0,
    length: float = 10e3,
    spacing: Optional[float] = None,
    parity: int = 1,
    port: Tuple[int, int] = (0, 0),
    direction: str = "WEST",
    waveguide_template: ComponentFactory = strip,
    layer: Layer = gf.LAYER.WG,
    layer_cladding: Layer = gf.LAYER.WGCLAD,
    cladding_offset: float = 3.0,
    wg_width: float = 0.5,
    radius: float = 10.0,
    **kwargs
) -> Component:
    """Picwriter Spiral

    Args:
        port_spacing: distance between input/output ports
        length: spiral length (um)
        spacing: distance between parallel straights
        parity: If 1 spiral on right side, if -1 spiral on left side (mirror flip)
        port: Cartesian coordinate of the input port
        direction: NORTH, WEST, SOUTH, EAST  or angle in radians
        waveguide_template (WaveguideTemplate): Picwriter WaveguideTemplate function
        layer: core layer
        layer_cladding: cladding layer
        cladding_offset: distance from core to cladding
        wg_width: 0.5
        radius: 10
        kwargs: cross_section settings

    """
    c = pc.Spiral(
        gf.call_if_func(
            waveguide_template,
            wg_width=wg_width,
            radius=radius,
            layer=layer,
            layer_cladding=layer_cladding,
            cladding_offset=cladding_offset,
            **kwargs
        ),
        width=port_spacing,
        length=length,
        spacing=spacing,
        parity=parity,
        port=port,
        direction=direction,
    )
    return gf.read.picwriter(c, port_layer=layer)


if __name__ == "__main__":
    c = spiral(length=10e3, port_spacing=500, radius=20, direction="NORTH")
    c = spiral(length=10e3, port_spacing=500, radius=20, direction="EAST")
    c.show()
