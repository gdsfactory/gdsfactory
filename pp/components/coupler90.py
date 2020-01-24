import pp
from pp import Component
from pp.components.waveguide import waveguide
from pp.components.bend_circular import bend_circular
from pp.name import autoname

__version__ = "0.0.1"


@autoname
def coupler90(bend_radius=10.0, width=0.5, gap=0.2):
    """ Waveguide coupled to a bend with gap

    Args:
        bend_radius: um
        width: waveguide width (um)
        gap: um

    .. plot::
      :include-source:

      import pp
      c = pp.c.coupler90()
      pp.plotgds(c)

    """
    y = width + gap
    _bend = bend_circular(radius=bend_radius, width=width).ref((0, y))
    c = Component()

    _wg = c.add_ref(waveguide(length=bend_radius, width=width))
    c.add(_bend)

    # This component is a leaf cell => using absorb
    c.absorb(_wg)
    c.absorb(_bend)

    port_width = 2 * width + gap

    c.add_port(port=_wg.ports["E0"], name="E0")
    c.add_port(port=_bend.ports["N0"], name="N0")
    c.add_port(name="W0", midpoint=[0, y / 2], width=port_width, orientation=180)
    c.y = y
    return c


def _demo():
    coupler = coupler90()
    return coupler


if __name__ == "__main__":
    c = _demo()
    pp.show(c)
