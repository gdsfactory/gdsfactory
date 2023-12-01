from __future__ import annotations

from collections.abc import Callable

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.taper_cross_section import taper_cross_section
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports_optical
from gdsfactory.typings import ComponentFactory, ComponentSpec, CrossSectionSpec


@cell
def add_tapers(
    component: ComponentSpec,
    taper: ComponentFactory | Component = taper_cross_section,
    select_ports: Callable | None = select_ports_optical,
    taper_port_name1: str = "o1",
    taper_port_name2: str = "o2",
    cross_section1: CrossSectionSpec | None = None,
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
        cross_section1: optional start cross_section factory. Defaults to port.
        cross_section2: end cross_section factory (cross_section).

    Keyword Args:
        length: transition length.
        npoints: number of points.
        linear: shape of the transition, sine when False.
        kwargs: cross_section settings for section2.

    Note:
        If ``taper`` is a partial function and ``cross_section2`` is None, then
        ``cross_section2`` is inferred from the partial keywords.
    """
    c = gf.Component()
    component = gf.get_component(component)

    ports_to_taper = select_ports(component.ports) if select_ports else component.ports
    ports_to_taper_names = [p.name for p in ports_to_taper.values()]

    for port_name, port in component.ports.items():
        if port.name in ports_to_taper_names:
            if (
                isinstance(taper, gf.partial)
                and "cross_section2" in taper.keywords
                and cross_section2 is None
            ):
                _taper = gf.get_component(
                    taper,
                    cross_section2=gf.partial(
                        taper.keywords["cross_section2"], width=port.width
                    ),
                )
            elif isinstance(taper, Component):
                _taper = taper
            else:
                _taper = gf.get_component(
                    taper,
                    cross_section1=cross_section1 or port.cross_section,
                    cross_section2=cross_section2,
                    **kwargs,
                )
            taper_ref = c << _taper
            taper_ref.connect(taper_ref.ports[taper_port_name1].name, port)
            c.add_port(name=port_name, port=taper_ref.ports[taper_port_name2])
        else:
            c.add_port(name=port_name, port=port)
    c.add_ref(component)
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c0 = gf.components.straight(width=2, cross_section=gf.cross_section.rib)
    xs_rc_tip = gf.cross_section.strip_rib_tip

    c1 = add_tapers(c0, cross_section2=xs_rc_tip, linear=True)
    c1.show()
