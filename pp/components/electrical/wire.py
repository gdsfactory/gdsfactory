from pp.name import autoname
from pp.component import Component
from pp.layers import LAYER
from pp.port import deco_rename_ports
from pp.components.hline import hline


WIRE_WIDTH = 10.0


@deco_rename_ports
@autoname
def wire(length=50.0, width=WIRE_WIDTH, layer=LAYER.M3):
    """ electrical straight wire

    .. plot::
      :include-source:

      import pp

      c = pp.c.wire(length=50., width=10., layer=pp.LAYER.M3)
      pp.plotgds(c)
    """
    return hline(length=length, width=width, layer=layer)


@deco_rename_ports
@autoname
def corner(width=WIRE_WIDTH, radius=None, layer=LAYER.M3):
    """ 90 degrees electrical bend

    Args:
        width: wire width
        radius ignore (passed for consistency with other types of bends)
        layer: layer

    .. plot::
      :include-source:

      import pp

      c = pp.c.corner(width=10., layer=pp.LAYER.M3)
      pp.plotgds(c)
    """
    c = Component()
    a = width / 2
    xpts = [-a, a, a, -a]
    ypts = [-a, -a, a, a]

    c.add_polygon([xpts, ypts], layer=layer)
    c.add_port(name="W0", midpoint=(-a, 0), width=width, orientation=180)
    c.add_port(name="N0", midpoint=(0, a), width=width, orientation=90)
    c.info["length"] = width
    return c


if __name__ == "__main__":
    import pp

    c = wire()
    pp.show(c)
