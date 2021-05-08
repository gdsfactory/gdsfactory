from typing import List, Tuple

from pp.port import Port


def get_port_x(port: Port) -> float:
    return port.midpoint[0]


def get_port_y(port: Port) -> float:
    return port.midpoint[1]


def sort_ports(ports1: List[Port], ports2: List[Port]) -> Tuple[List[Port], List[Port]]:
    """Returns sorted ports."""

    if ports1[0].orientation in [0, 180] and ports2[0].orientation in [0, 180]:
        f_key1 = get_port_y
        f_key2 = get_port_y
        ports1.sort(key=f_key1)
        ports2.sort(key=f_key2)
    elif ports1[0].orientation in [90, 270] and ports2[0].orientation in [90, 270]:
        f_key1 = get_port_x
        f_key2 = get_port_x
        ports1.sort(key=f_key1)
        ports2.sort(key=f_key2)
    else:
        if ports1[0].orientation in [0, 180]:
            axis = "X"
        else:
            axis = "Y"

        if axis in ["X", "x"]:
            f_key1 = get_port_y
            # f_key2 = get_port_y
        else:
            f_key1 = get_port_x
            # f_key2 = get_port_x

        ports2_by1 = {p1: p2 for p1, p2 in zip(ports1, ports2)}
        ports1.sort(key=f_key1)
        ports2 = [ports2_by1[p1] for p1 in ports1]
    return ports1, ports2
