from __future__ import annotations

import kfactory as kf

from gdsfactory.port import Port


def get_port_x(port: Port) -> float:
    return port.dcenter[0]


def get_port_y(port: Port) -> float:
    return port.dcenter[1]


def sort_ports_x(ports: list[Port]) -> list[Port]:
    ports = list(ports)
    f_key = get_port_x
    ports.sort(key=f_key)
    return ports


def sort_ports_y(ports: list[Port]) -> list[Port]:
    ports = list(ports)
    f_key = get_port_y
    ports.sort(key=f_key)
    return ports


def sort_ports(
    ports1: list[Port] | kf.Ports,
    ports2: list[Port] | kf.Ports,
    enforce_port_ordering: bool,
) -> tuple[list[Port], list[Port]]:
    """Returns two lists of sorted ports.

    Args:
        ports1: the starting ports
        ports2: the ending ports
        enforce_port_ordering: if True, only ports2 will be sorted in accordance with ports1.
            If False, the two lists will be sorted independently.

    """
    ports1 = list(ports1)
    ports2 = list(ports2)

    if len(ports1) != len(ports2):
        raise ValueError(f"ports1={len(ports1)} and ports2={len(ports2)} must be equal")
    if not ports1:
        raise ValueError("ports1 is an empty list")
    if not ports2:
        raise ValueError("ports2 is an empty list")

    if ports1[0].orientation in [0, 180] and ports2[0].orientation in [0, 180]:
        _sort(get_port_y, ports1, enforce_port_ordering, ports2)
    elif ports1[0].orientation in [90, 270] and ports2[0].orientation in [90, 270]:
        _sort(get_port_x, ports1, enforce_port_ordering, ports2)
    else:
        axis = "X" if ports1[0].orientation in [0, 180] else "Y"
        f_key1 = get_port_y if axis in {"X", "x"} else get_port_x
        ports1.sort(key=f_key1)
        if not enforce_port_ordering:
            ports2.sort(key=f_key1)

    if enforce_port_ordering:
        ports2 = [ports2[ports1.index(p1)] for p1 in ports1]

    return ports1, ports2


def _sort(key_func, ports1, enforce_port_ordering, ports2):
    ports1.sort(key=key_func)
    if not enforce_port_ordering:
        ports2.sort(key=key_func)


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    c1 = c << gf.c.straight()
    c2 = c << gf.c.straight()
    sort_ports(c1.ports, c2.ports, enforce_port_ordering=True)
    c.show()
