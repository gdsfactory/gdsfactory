from typing import Optional

import pp
from pp.components.bend_euler import bend_euler, bend_euler_s
from pp.components.mmi1x2 import mmi1x2
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory


@pp.cell
def splitter_tree(
    coupler: ComponentFactory = mmi1x2,
    noutputs: int = 4,
    spacing: Optional[float] = None,
    spacing_extra: float = 0.1,
    bend_factory: ComponentFactory = bend_euler,
    bend_s: ComponentFactory = bend_euler_s,
    bend_radius: Optional[float] = None,
    tech: Optional[Tech] = None,
    auto_widen: bool = True,
) -> pp.Component:
    """Tree of 1x2 splitters

    Args:
        coupler: 1x2 coupler factory
        noutputs:
        spacing: y spacing for outputs
        spacing_extra: extra y spacing for outputs
        bend_s: factory for terminating ports
        bend_factory:
        bend_radius: for routing, defaults to tech.bend_radius
        tech: technology
        auto_widen: tapers to lower loss wide straights.

    .. code::

             __|
          __|  |__
        _|  |__
         |__       spacing


    """
    tech = tech or TECH_SILICON_C
    bend_radius = bend_radius or tech.bend_radius
    bend90 = bend_factory(radius=bend_radius)
    c = pp.Component()

    coupler = pp.call_if_func(coupler)
    coupler_ref = c.add_ref(coupler)
    dy = spacing if spacing else bend90.dy * noutputs + spacing_extra
    dx = coupler.xsize

    if noutputs > 2:
        c2 = splitter_tree(
            coupler=coupler,
            noutputs=noutputs // 2,
            bend_radius=bend_radius,
            spacing=dy / 2,
            bend_factory=bend_factory,
            tech=tech,
        )
    else:
        c2 = bend_s(radius=dy / 2)

    if dy < 3 * bend_radius:
        tree_top = c2.ref(port_id="W0", position=coupler_ref.ports["E1"])
        tree_bot = c2.ref(port_id="W0", position=coupler_ref.ports["E0"], v_mirror=True)

    else:
        tree_top = c2.ref(
            port_id="W0", position=coupler_ref.ports["E1"].position + (dx, dy)
        )
        tree_bot = c2.ref(
            port_id="W0",
            position=coupler_ref.ports["E0"].position + (dx, -dy),
            v_mirror=False,  # True,
        )

        c.add(
            pp.routing.get_route(
                coupler.ports["E1"],
                tree_top.ports["W0"],
                bend_radius=bend_radius,
                bend_factory=bend_factory,
                auto_widen=auto_widen,
            )["references"]
        )
        c.add(
            pp.routing.get_route(
                coupler.ports["E0"],
                tree_bot.ports["W0"],
                bend_radius=bend_radius,
                bend_factory=bend_factory,
                auto_widen=auto_widen,
            )["references"]
        )

    i = 0
    for p in pp.port.get_ports_facing(tree_bot, "E"):
        c.add_port(name=f"E{i}", port=p)
        i += 1

    for p in pp.port.get_ports_facing(tree_top, "E"):
        c.add_port(name=f"E{i}", port=p)
        i += 1

    c.add(tree_bot)
    c.add(tree_top)
    c.add_port(name="W0", port=coupler_ref.ports["W0"])
    return c


if __name__ == "__main__":
    c = splitter_tree(coupler=pp.components.mmi1x2(), noutputs=50)
    c.show()
    print(c.get_ports_dict().keys())
    print(c.spacing)
