from __future__ import annotations

import kfactory as kf

from gdsfactory.port import Port


def get_port_x(port: Port) -> float:
    return port.d.center[0]


def get_port_y(port: Port) -> float:
    return port.d.center[1]


def sort_ports_x(ports: list[Port]) -> list[Port]:
    return sorted(ports, key=get_port_x)


def sort_ports_y(ports: list[Port]) -> list[Port]:
    return sorted(ports, key=get_port_y)


def sort_ports(
    ports1: list[Port] | kf.Ports,
    ports2: list[Port] | kf.Ports,
    enforce_port_ordering: bool = True,
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

    # Determine the sorting axis based on orientation
    if ports1[0].orientation in [0, 180]:
        ports1_sorted = sort_ports_y(ports1)
        ports2_sorted = sort_ports_y(ports2)
    else:
        ports1_sorted = sort_ports_x(ports1)
        ports2_sorted = sort_ports_x(ports2)

    # Enforce port ordering if required
    if enforce_port_ordering:
        ports2_sorted = [ports2_sorted[ports1_sorted.index(p1)] for p1 in ports1_sorted]

    return ports1_sorted, ports2_sorted


def handle_flipped_orientations(ports1, ports2):
    # Separate ports by their orientations
    ports1_0_180 = [port for port in ports1 if port.orientation in [0, 180]]
    ports2_0_180 = [port for port in ports2 if port.orientation in [0, 180]]
    ports1_90_270 = [port for port in ports1 if port.orientation in [90, 270]]
    ports2_90_270 = [port for port in ports2 if port.orientation in [90, 270]]

    # Sort each group of ports
    ports1_sorted_0_180, ports2_sorted_0_180 = sort_ports(ports1_0_180, ports2_0_180)
    ports1_sorted_90_270, ports2_sorted_90_270 = sort_ports(
        ports1_90_270, ports2_90_270
    )

    # Combine sorted ports back together
    sorted_ports1 = ports1_sorted_0_180 + ports1_sorted_90_270
    sorted_ports2 = ports2_sorted_0_180 + ports2_sorted_90_270

    return sorted_ports1, sorted_ports2
