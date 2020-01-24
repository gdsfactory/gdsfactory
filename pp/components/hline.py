import pp
from pp.layers import LAYER
from pp.name_and_cache import autoname_and_cache

__version__ = "0.0.1"


@autoname_and_cache
def hline(length=10, width=0.5, layer=LAYER.WG):
    """ horizonal line waveguide, with ports on east and west sides

    .. plot::
      :include-source:

      import pp
      c = pp.c.hline()
      pp.plotgds(c)

    """
    c = pp.Component()
    a = width / 2
    if length > 0 and width > 0:
        c.add_polygon([(0, -a), (length, -a), (length, a), (0, a)], layer=layer)

    c.add_port(name="W0", midpoint=[0, 0], width=width, orientation=180, layer=layer)
    c.add_port(name="E0", midpoint=[length, 0], width=width, orientation=0, layer=layer)

    c.width = width
    c.length = length
    return c


if __name__ == "__main__":
    c = hline(width=10)
    print(c)
    pp.show(c)
