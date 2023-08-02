from __future__ import annotations

from collections.abc import Callable

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports_optical
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@cell
def add_tapers(
    component: Component,
    taper: ComponentSpec = taper_cross_section,
    select_ports: Callable | None = select_ports_optical,
    taper_port_name1: str = "o1",
    taper_port_name2: str = "o2",
    cross_section2: CrossSectionSpec = strip,
    **kwargs,
) -> Component:
    """Returns new component with taper in all optical ports.

    Args:
        component: to add tapers.
        taper: taper spec.
        select_ports: function to select ports.
        taper_port_name1: name.
        taper_port_name2: name.
        cross_section2: end cross_section factory (cross_section).

    Keyword Args:
        cross_section1: start cross_section factory.
        length: transition length.
        npoints: number of points.
        linear: shape of the transition, sine when False.
        kwargs: cross_section settings for section2.
    """
    c = gf.Component()
    ports_to_taper = select_ports(component.ports) if select_ports else component.ports
    ports_to_taper_names = [p.name for p in ports_to_taper.values()]

    for port_name, port in component.ports.items():
        if port.name in ports_to_taper_names:
            taper_ref = c << taper(
                cross_section1=port.cross_section,
                cross_section2=cross_section2,
                **kwargs,
            )
            taper_ref.connect(taper_ref.ports[taper_port_name1].name, port)
            c.add_port(name=port_name, port=taper_ref.ports[taper_port_name2])
        else:
            c.add_port(name=port_name, port=port)
    c.add_ref(component)
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c0 = gf.components.straight(width=2, cross_section=gf.cross_section.rib)
    xs_rib_tip = gf.cross_section.strip_rib_tip

    c1 = add_tapers(c0, cross_section2=xs_rib_tip, linear=True)
    c1.show()
