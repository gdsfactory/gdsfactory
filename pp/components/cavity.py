import pp
from pp.components.coupler import coupler

# from pp.components.loop_mirror import loop_mirror
from pp.container import container
from pp.component import Component


@container
@pp.autoname
def cavity(
    component: Component,
    coupler: Component = coupler,
    length: float = 0.1,
    gap: float = 0.2,
    wg_width: float = 0.5,
) -> Component:
    """ creates a cavity from a coupler and a mirror
    it will connect the W0 port of the mirror to both E1 and W1 ports of the coupler creating a resonant cavity

    Args:
        component: mirror
        coupler: coupler factory
        length: coupler length
        gap: coupler gap
        wg_width: coupler wg_width

    .. code::

      ml (mirror left)              mr (mirror right)
       |                               |
       |W0 - W1__             __E1 - W0|
       |         \           /         |
                  \         /
                ---=========---
             W0    length      E0

    .. plot::
      :include-source:

      import pp

      c = pp.c.cavity(component=pp.c.dbr())
      pp.plotgds(c)
    """
    mirror = pp.call_if_func(component)
    coupler = pp.call_if_func(coupler, length=length, gap=gap, wg_width=wg_width)

    c = pp.Component()
    cr = c << coupler
    ml = c << mirror
    mr = c << mirror

    ml.connect("W0", destination=cr.ports["W1"])
    mr.connect("W0", destination=cr.ports["E1"])
    c.add_port("W0", port=cr.ports["W0"])
    c.add_port("E0", port=cr.ports["E0"])
    return c


if __name__ == "__main__":
    from pp.components.dbr import dbr

    c = cavity(component=dbr(), pins=True)
    pp.show(c)
