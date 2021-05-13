from typing import Optional

from pydantic import validate_arguments

import pp
from pp.cell import cell
from pp.components.bend_euler import bend_euler, bend_euler_s
from pp.components.mmi1x2 import mmi1x2
from pp.cross_section import get_cross_section_settings
from pp.types import ComponentFactory, ComponentOrFactory


@cell
@validate_arguments
def splitter_tree(
    coupler: ComponentOrFactory = mmi1x2,
    noutputs: int = 4,
    spacing: Optional[float] = None,
    spacing_extra: float = 0.1,
    bend_factory: ComponentFactory = bend_euler,
    bend_s: ComponentFactory = bend_euler_s,
    cross_section_name: str = "strip",
    **kwargs,
) -> pp.Component:
    """Tree of 1x2 splitters

    Args:
        coupler: 1x2 coupler factory
        noutputs:
        spacing: y spacing for outputs
        spacing_extra: extra y spacing for outputs
        bend_factory:
        bend_s: factory for terminating ports

    .. code::

             __|
          __|  |__
        _|  |__
         |__       spacing


    """
    cross_section_settings = get_cross_section_settings(cross_section_name, **kwargs)
    radius = cross_section_settings.get("radius")

    bend90 = bend_factory(radius=radius)
    c = pp.Component()

    coupler = pp.call_if_func(
        coupler, cross_section_name=cross_section_name, **cross_section_settings
    )
    coupler_ref = c.add_ref(coupler)
    dy = spacing if spacing else bend90.dy * noutputs + spacing_extra
    dx = coupler.xsize + radius

    if noutputs > 2:
        c2 = splitter_tree(
            coupler=coupler,
            noutputs=noutputs // 2,
            spacing=dy / 2,
            bend_factory=bend_factory,
            cross_section_name=cross_section_name,
            spacing_extra=spacing_extra,
            **cross_section_settings,
        )
    else:
        c2 = bend_s(radius=dy / 2, cross_section_name=cross_section_name)

    if dy < 3 * radius:
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
                bend_factory=bend_factory,
                **cross_section_settings,
            )["references"]
        )
        c.add(
            pp.routing.get_route(
                coupler.ports["E0"],
                tree_bot.ports["W0"],
                bend_factory=bend_factory,
                **cross_section_settings,
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
    c.dy = dy
    return c


if __name__ == "__main__":
    c = splitter_tree(
        coupler=pp.components.mmi1x2,
        noutputs=128 * 2,
        cross_section_name="nitride",
    )
    print(len(c.ports))
    c.show()
