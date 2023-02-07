from __future__ import annotations

from typing import Dict, List, Optional, Union

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.cross_section import strip
from gdsfactory.difftest import difftest
from gdsfactory.port import Port
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Routes


def get_routes_bend180(
    ports: Union[List[Port], Dict[str, Port]],
    bend: ComponentSpec = bend_euler,
    cross_section: CrossSectionSpec = strip,
    bend_port1: Optional[str] = None,
    bend_port2: Optional[str] = None,
    **kwargs,
) -> Routes:
    """Returns routes made by 180 degree bends.

    Args:
        ports: List or dict of ports.
        bend: function for bend.
        cross_section: spec.
        bend_port1: name.
        bend_port2: name.
        kwargs: bend settings.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component("get_routes_bend180")
        pad_array = gf.components.pad_array(orientation=270)
        c1 = c << pad_array
        c2 = c << pad_array
        c2.rotate(90)
        c2.movex(1000)
        c2.ymax = -200

        routes_bend180 = gf.routing.get_routes_bend180(
            ports=c2.get_ports_list(), radius=75 / 2,
        )
        c.add(routes_bend180.references)

        routes = gf.routing.get_bundle(
            c1.get_ports_list(), routes_bend180.ports,
        )
        for route in routes:
            c.add(route.references)
        c.show(show_ports=True)
        c.plot()

    """
    ports = list(ports.values()) if isinstance(ports, dict) else ports
    bend = bend(angle=180, cross_section=cross_section, **kwargs)

    bend_ports = bend.get_ports_list()
    bend_port1 = bend_port1 or bend_ports[0].name
    bend_port2 = bend_port2 or bend_ports[1].name

    references = [bend.ref() for _ in ports]
    references = [ref.connect(bend_port1, port) for port, ref in zip(ports, references)]
    ports = [ref.ports[bend_port2] for ref in references]
    lengths = [bend.info["length"]] * len(ports)
    return Routes(references=references, ports=ports, lengths=lengths)


def test_get_routes_bend180():
    c = gf.Component("get_routes_bend180")
    pad_array = gf.components.pad_array(orientation=270)
    c1 = c << pad_array
    c2 = c << pad_array
    c2.rotate(90)
    c2.movex(1000)
    c2.ymax = -200

    routes_bend180 = get_routes_bend180(
        ports=c2.get_ports_list(),
        radius=75 / 2,
    )
    c.add(routes_bend180.references)

    routes = gf.routing.get_bundle(
        ports1=c1.get_ports_list(),
        ports2=routes_bend180.ports,
    )
    for route in routes:
        c.add(route.references)
    difftest(c)
    return c


if __name__ == "__main__":
    c = gf.Component("get_routes_bend180")
    pad_array = gf.components.pad_array(orientation=270)
    c1 = c << pad_array
    c2 = c << pad_array
    c2.rotate(90)
    c2.movex(1000)
    c2.ymax = -200
    layer = (2, 0)
    routes_bend180 = get_routes_bend180(
        ports=c2.get_ports_list(), radius=75 / 2, layer=layer
    )
    c.add(routes_bend180.references)

    routes = gf.routing.get_bundle(
        c1.get_ports_list(), routes_bend180.ports, layer=layer
    )
    for route in routes:
        c.add(route.references)
    c.show(show_ports=True)
