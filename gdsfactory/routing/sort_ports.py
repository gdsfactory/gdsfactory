from __future__ import annotations

from collections.abc import Callable

from gdsfactory.typings import Port, Ports


def get_port_x(port: Port) -> float:
    return port.center[0]


def get_port_y(port: Port) -> float:
    return port.center[1]


def sort_ports_x(ports: Ports) -> list[Port]:
    ports = list(ports)
    f_key = get_port_x
    ports.sort(key=f_key)
    return ports


def sort_ports_y(ports: Ports) -> list[Port]:
    ports = list(ports)
    f_key = get_port_y
    ports.sort(key=f_key)
    return ports


def sort_ports(
    ports1: Ports,
    ports2: Ports,
    enforce_port_ordering: bool,
) -> tuple[list[Port], list[Port]]:
    """Returns two lists of sorted ports.

    Args:
        ports1: the starting ports
        ports2: the ending ports
        enforce_port_ordering: if True, only ports2 will be sorted in accordance with ports1.
            If False, the two lists will be sorted independently.

    """
    original_ports1 = list(ports1)
    ports1_list = list(ports1)
    ports2_list = list(ports2)

    if len(ports1_list) != len(ports2_list):
        raise ValueError(
            f"ports1={len(ports1_list)} and ports2={len(ports2_list)} must be equal"
        )
    if not ports1_list:
        raise ValueError("ports1 is an empty list")

    if ports1_list[0].orientation in [0, 180] and ports2_list[0].orientation in [
        0,
        180,
    ]:
        _sort(get_port_y, ports1_list, enforce_port_ordering, ports2_list)
    elif ports1_list[0].orientation in [90, 270] and ports2_list[0].orientation in [
        90,
        270,
    ]:
        _sort(get_port_x, ports1_list, enforce_port_ordering, ports2_list)
    else:
        axis = "X" if ports1_list[0].orientation in [0, 180] else "Y"
        f_key1 = get_port_y if axis == "X" else get_port_x
        ports1_list.sort(key=f_key1)
        if not enforce_port_ordering:
            ports2_list.sort(key=f_key1)

    if enforce_port_ordering:
        ports2_list = [ports2_list[original_ports1.index(p1)] for p1 in ports1_list]

    return ports1_list, ports2_list


def _sort(
    key_func: Callable[[Port], float],
    ports1: list[Port],
    enforce_port_ordering: bool,
    ports2: list[Port],
) -> None:
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
