from typing import List, Tuple

from pp.port import Port


def get_port_x(port: Port) -> float:
    return port.midpoint[0]


def get_port_y(port: Port) -> float:
    return port.midpoint[1]


def sort_ports(ports1: List[Port], ports2: List[Port]) -> Tuple[List[Port], List[Port]]:
    """Returns sorted ports."""
    if ports1[0].angle in [0, 180]:
        # axis = "X"
        f_key1 = get_port_y
        f_key2 = get_port_y
    else:
        # axis = "Y"
        f_key1 = get_port_x
        f_key2 = get_port_x

    ports1.sort(key=f_key1)
    ports2.sort(key=f_key2)
    return ports1, ports2
