from typing import Callable, List, Tuple
from pp.name import autoname
from pp.layers import LAYER
from pp.components.compass import compass
from pp.component import Component

WIRE_WIDTH = 10.0


@autoname
def pad(
    width: int = 100, height: int = 100, layer: Tuple[int, int] = LAYER.M3
) -> Component:
    """rectangular pad with 4 ports (N, S, E, W)

    Args:
        width: pad width
        height: pad


    .. plot::
      :include-source:

      import pp

      c = pp.c.pad(width=100, height=100, layer=pp.LAYER.M3)
      pp.plotgds(c)

    """
    c = Component()
    _c = compass(size=(width, height), layer=layer).ref()
    c.add(_c)
    c.absorb(_c)
    c.ports = _c.ports
    return c


@autoname
def pad_array(
    pad: Callable = pad,
    start: Tuple[int, int] = (0, 0),
    spacing: Tuple[int, int] = (150, 0),
    n: int = 6,
    port_list: List[str] = ["N"],
    width: int = 100,
    height: int = 100,
    layer: Tuple[int, int] = LAYER.M3,
) -> Component:
    """array of rectangular pads

    Args:
        pad: pad element
        start: start coordinate
        spacing: (x, y) spacing
        n: number of pads
        port_list: list of port orientations (N, S, W, E) per pad

    .. plot::
      :include-source:

      import pp

      c = pp.c.pad_array(pad=pp.c.pad, start=(0, 0), spacing=(150, 0), n=6, port_list=["N"])
      pp.plotgds(c)

    """
    c = Component()
    pad = pad(width=width, height=height, layer=layer) if callable(pad) else pad

    for i in range(n):
        p = c << pad
        p.x = i * spacing[0]
        for port_name in port_list:
            port_name_new = "{}{}".format(port_name, i)
            c.add_port(port=p.ports[port_name], name=port_name_new)

    c.move(start)
    return c


if __name__ == "__main__":
    import pp

    # c = pad()
    # c = pad(width=10, height=10)
    # print(c.ports.keys())
    # print(c.settings['spacing'])
    c = pad_array()
    pp.show(c)
