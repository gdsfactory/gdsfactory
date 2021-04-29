from typing import List, Tuple

from pp.cell import cell
from pp.component import Component
from pp.components.compass import compass
from pp.layers import LAYER
from pp.types import ComponentOrFactory

WIRE_WIDTH = 10.0


@cell
def pad(
    width: int = 100, height: int = 100, layer: Tuple[int, int] = LAYER.M3
) -> Component:
    """rectangular pad with 4 ports (N, S, E, W)

    Args:
        width: pad width
        height: pad height
        layer: pad layer

    """
    c = Component()
    _c = compass(size=(width, height), layer=layer).ref()
    c.add(_c)
    c.absorb(_c)
    c.ports = _c.ports
    return c


@cell
def pad_array(
    pad: ComponentOrFactory = pad,
    pitch: float = 150.0,
    n: int = 6,
    port_list: List[str] = ("N",),
    **pad_settings,
) -> Component:
    """array of rectangular pads

    Args:
        pad: pad element
        pitch: x spacing
        n: number of pads
        port_list: list of port orientations (N, S, W, E) per pad
        pad_settings: settings for pad if pad is callable


    """
    c = Component()
    pad = pad(**pad_settings) if callable(pad) else pad

    for i in range(n):
        p = c << pad
        p.x = i * pitch
        for port_name in port_list:
            port_name_new = f"{port_name}{i}"
            c.add_port(port=p.ports[port_name], name=port_name_new)

    return c


if __name__ == "__main__":

    c = pad()
    print(c.ports)
    # c = pad(width=10, height=10)
    # print(c.ports.keys())
    # print(c.settings['spacing'])
    c = pad_array()
    c.show()
