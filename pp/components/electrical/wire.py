import pp
from pp.layers import LAYER
from pp.ports.port_naming import deco_rename_ports
from pp.components.hline import hline

__version__ = "0.0.1"

WIRE_WIDTH = 10.0


@deco_rename_ports
@pp.autoname
def wire(length=50, width=WIRE_WIDTH, layer=LAYER.M3):
    """
    straight wire
    """
    return hline(length=length, width=width, layer=layer)


@deco_rename_ports
@pp.autoname
def corner(width=WIRE_WIDTH, radius=None, layer=LAYER.M3):
    """
    radius passed for consistency with other types of bends
    """
    component = pp.Component()
    a = width / 2
    xpts = [-a, a, a, -a]
    ypts = [-a, -a, a, a]

    component.add_polygon([xpts, ypts], layer=layer)
    component.add_port(name="W0", midpoint=(-a, 0), width=width, orientation=180)
    component.add_port(name="N0", midpoint=(0, a), width=width, orientation=90)
    component.info["length"] = width

    return component


def test_wire():
    c = wire()
    pp.write_gds(c)
    return c


if __name__ == "__main__":
    test_wire()
