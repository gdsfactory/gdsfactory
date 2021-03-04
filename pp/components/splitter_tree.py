from typing import Optional

import pp
from pp.components.bend_euler import bend_euler
from pp.components.mmi1x2 import mmi1x2
from pp.components.waveguide import waveguide
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory


@pp.cell
def splitter_tree(
    coupler: ComponentFactory = mmi1x2,
    noutputs: int = 4,
    spacing: Optional[float] = None,
    spacing_extra: float = 0.1,
    terminator: ComponentFactory = waveguide,
    bend_factory: ComponentFactory = bend_euler,
    bend_radius: Optional[float] = None,
    tech: Tech = TECH_SILICON_C,
) -> pp.Component:
    """tree of 1x2 splitters

    Args:
        coupler: 1x2 coupler factory
        noutputs:
        bend_radius: for routing
        spacing: 2X spacing
        termination: factory for terminating ports

    .. code::

             __|
          __|  |__
        _|  |__
         |__       spacing


    .. plot::
      :include-source:

      import pp

      c = pp.c.splitter_tree(coupler=pp.c.mmi1x2(), noutputs=4, spacing=50, bend_radius=10)
      c.plot()

    """
    bend_radius = bend_radius or tech.bend_radius
    bend90 = bend_factory(radius=bend_radius)
    noutputs = noutputs
    c = pp.Component()

    coupler = pp.call_if_func(coupler)
    _coupler = c.add_ref(coupler)
    coupler_sep = coupler.ports["E1"].y - coupler.ports["E0"].y
    spacing = spacing if spacing else bend90.dy * noutputs + spacing_extra

    if noutputs > 2:
        c2 = splitter_tree(
            coupler=coupler,
            noutputs=noutputs // 2,
            bend_radius=bend_radius,
            spacing=spacing / 2,
            bend_factory=bend_factory,
            tech=tech,
        )
    else:
        c2 = terminator()

    if spacing < coupler_sep:
        tree_top = c2.ref(port_id="W0", position=_coupler.ports["E1"])
        tree_bot = c2.ref(
            port_id="W0", position=_coupler.ports["E0"], v_mirror=False  # True
        )

    else:
        dx = dy = spacing
        tree_top = c2.ref(
            port_id="W0", position=_coupler.ports["E1"].position + (dx, dy)
        )
        tree_bot = c2.ref(
            port_id="W0",
            position=_coupler.ports["E0"].position + (dx, -dy),
            v_mirror=False,  # True,
        )

        c.add(
            pp.routing.get_route(
                coupler.ports["E1"],
                tree_top.ports["W0"],
                bend_radius=bend_radius,
                bend_factory=bend_factory,
            )["references"]
        )
        c.add(
            pp.routing.get_route(
                coupler.ports["E0"],
                tree_bot.ports["W0"],
                bend_radius=bend_radius,
                bend_factory=bend_factory,
            )["references"]
        )

    i = 0
    for p in pp.port.get_ports_facing(tree_bot, "E"):
        c.add_port(name=f"{i}", port=p)
        i += 1

    for p in pp.port.get_ports_facing(tree_top, "E"):
        c.add_port(name=f"{i}", port=p)
        i += 1

    c.add(tree_bot)
    c.add(tree_top)
    c.add_port(name="W0", port=_coupler.ports["W0"])

    return c


if __name__ == "__main__":
    c = splitter_tree(coupler=pp.c.mmi1x2(), noutputs=20)
    c.show()
