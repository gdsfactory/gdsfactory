from typing import Dict, List, Union

import gdsfactory as gf
from gdsfactory.components.straight import straight
from gdsfactory.difftest import difftest
from gdsfactory.port import Port
from gdsfactory.types import ComponentOrFactory, Routes


def get_routes_straight(
    ports: Union[List[Port], Dict[str, Port]],
    straight: ComponentOrFactory = straight,
    **kwargs,
) -> Routes:
    """Returns routes made by 180 degree straights.

    Args:
        ports: List or dict of ports
        straight: function for straight
        **kwargs: waveguide settings
    """
    ports = list(ports.values()) if isinstance(ports, dict) else ports
    straight = straight(**kwargs)
    references = [straight.ref() for port in ports]
    references = [ref.connect("o1", port) for port, ref in zip(ports, references)]
    ports = [ref.ports["o2"] for i, ref in enumerate(references)]
    lengths = [straight.info.length] * len(ports)
    return Routes(references=references, ports=ports, lengths=lengths)


def test_get_routes_straight(check: bool = True):
    c = gf.Component("get_routes_straight")
    pad_array = gf.components.pad_array()
    c1 = c << pad_array
    c2 = c << pad_array
    c2.ymax = -200

    routes = get_routes_straight(ports=c1.get_ports_list(), length=200)
    c.add(routes.references)
    if check:
        difftest(c)
    return c


if __name__ == "__main__":
    c = test_get_routes_straight(False)
    c.show()
