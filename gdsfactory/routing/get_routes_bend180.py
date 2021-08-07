from typing import Dict, List, Union

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.difftest import difftest
from gdsfactory.port import Port
from gdsfactory.types import ComponentOrFactory, Routes


def get_routes_bend180(
    ports: Union[List[Port], Dict[str, Port]],
    bend_factory: ComponentOrFactory = bend_euler,
    waveguide: str = "strip",
    **kwargs,
) -> Routes:
    """Returns routes made by 180 degree bends.

    Args:
        ports: List or dict of ports
        bend_factory: function for bend
        **kwargs: bend settings
    """
    ports = list(ports.values()) if isinstance(ports, dict) else ports
    bend = bend_factory(angle=180, waveguide=waveguide, **kwargs)
    references = [bend.ref() for port in ports]
    references = [ref.connect("W0", port) for port, ref in zip(ports, references)]
    ports = {f"{i}": ref.ports["W1"] for i, ref in enumerate(references)}
    lengths = [bend.length] * len(ports)
    return Routes(references=references, ports=ports, lengths=lengths)


def test_get_routes_bend180():
    c = gf.Component("get_routes_bend180")
    pad_array = gf.components.pad_array(pitch=150, port_list=("S",))
    c1 = c << pad_array
    c2 = c << pad_array
    c2.rotate(90)
    c2.movex(1000)
    c2.ymax = -200

    routes_bend180 = get_routes_bend180(
        ports=c2.get_ports_list(), radius=75 / 2, waveguide="metal_routing"
    )
    c.add(routes_bend180.references)

    routes = gf.routing.get_bundle(
        c1.get_ports_list(), routes_bend180.ports, waveguide="metal_routing"
    )
    for route in routes:
        c.add(route.references)
    difftest(c)
    return c


if __name__ == "__main__":
    c = gf.Component("get_routes_bend180")
    pad_array = gf.components.pad_array(pitch=150, port_list=("S",))
    c1 = c << pad_array
    c2 = c << pad_array
    c2.rotate(90)
    c2.movex(1000)
    c2.ymax = -200

    routes_bend180 = get_routes_bend180(
        ports=c2.get_ports_list(), radius=75 / 2, waveguide="metal_routing"
    )
    c.add(routes_bend180.references)

    routes = gf.routing.get_bundle(
        c1.get_ports_list(), routes_bend180.ports, waveguide="metal_routing"
    )
    for route in routes:
        c.add(route.references)
    c.show()
