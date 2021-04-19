from typing import Optional, Tuple

import picwriter.components as pc

import pp
from pp.component import Component
from pp.components.waveguide_template import wg_strip
from pp.picwriter_to_component import picwriter_to_component
from pp.port import auto_rename_ports
from pp.types import ComponentFactory, Layer


@pp.cell
def spiral(
    width: float = 500.0,
    length: float = 10e3,
    spacing: Optional[float] = None,
    parity: int = 1,
    port: Tuple[int, int] = (0, 0),
    direction: str = "NORTH",
    waveguide_template: ComponentFactory = wg_strip,
    layer: Layer = pp.LAYER.WG,
    layer_cladding: Layer = pp.LAYER.WGCLAD,
    cladding_offset: float = 3.0,
    wg_width: float = 0.5,
    bend_radius: float = 10.0,
) -> Component:
    """Picwriter Spiral

    Args:
       width: width of the spiral (i.e. distance between input/output ports)
       length: desired length of the straight (um)
       spacing: distance between parallel straights
       parity: If 1 spiral on right side, if -1 spiral on left side (mirror flip)
       port: Cartesian coordinate of the input port
       direction: Direction that the component will point *towards*, can be of type `'NORTH'`, `'WEST'`, `'SOUTH'`, `'EAST'`, OR an angle (float, in radians)
       waveguide_template (WaveguideTemplate): Picwriter WaveguideTemplate function
       layer: core layer
       layer_cladding: cladding layer
       cladding_offset: distance from core to cladding
       wg_width: 0.5
       bend_radius: 10

    """
    c = pc.Spiral(
        pp.call_if_func(
            waveguide_template,
            wg_width=wg_width,
            bend_radius=bend_radius,
            layer=layer,
            layer_cladding=layer_cladding,
            cladding_offset=cladding_offset,
        ),
        width=width,
        length=length,
        spacing=spacing,
        parity=parity,
        port=port,
        direction=direction,
    )
    # print(f'length = {length/1e4:.2f}cm')
    c = picwriter_to_component(c)
    c = auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = spiral(length=10e3, width=500, bend_radius=20)
    c.show()
