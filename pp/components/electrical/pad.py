from typing import List, Tuple

from pp.cell import cell
from pp.component import Component
from pp.components.compass import compass
from pp.layers import LAYER
from pp.types import ComponentFactory

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
    pad: ComponentFactory = pad,
    spacing: Tuple[int, int] = (150.0, 0.0),
    n: int = 6,
    port_list: List[str] = ("N",),
    width: float = 100.0,
    height: float = 100.0,
    layer: Tuple[int, int] = LAYER.M3,
) -> Component:
    """array of rectangular pads

    Args:
        pad: pad element
        spacing: (x, y) spacing
        n: number of pads
        port_list: list of port orientations (N, S, W, E) per pad
        width: pad width
        height: pad height
        layer: pad layer

    """
    c = Component()
    pad = pad(width=width, height=height, layer=layer) if callable(pad) else pad

    for i in range(n):
        p = c << pad
        p.x = i * spacing[0]
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
