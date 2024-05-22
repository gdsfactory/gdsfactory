from __future__ import annotations

import kfactory as kf
import networkx as nx

from gdsfactory.port import Port


def get_port_x(port: Port) -> float:
    return port.d.center[0]


def get_port_y(port: Port) -> float:
    return port.d.center[1]


def sort_ports_x(ports: list[Port]) -> list[Port]:
    return sorted(ports, key=get_port_x)


def sort_ports_y(ports: list[Port]) -> list[Port]:
    return sorted(ports, key=get_port_y)


def _create_bipartite_graph(ports1: list[Port], ports2: list[Port], axis: str):
    B = nx.Graph()
    for i, p1 in enumerate(ports1):
        for j, p2 in enumerate(ports2):
            weight = abs(p1.d.y - p2.d.y) if axis == "X" else abs(p1.d.x - p2.d.x)
            B.add_edge(f"p1_{i}", f"p2_{j}", weight=weight)
    return B


def _sort_ports_using_bipartite_matching(
    ports1: list[Port], ports2: list[Port], axis: str
):
    B = _create_bipartite_graph(ports1, ports2, axis)
    matching = nx.max_weight_matching(B, maxcardinality=True)

    sorted_ports1 = []
    sorted_ports2 = []
    for p1_node, p2_node in sorted(matching):
        p1_idx = int(p1_node.split("_")[1])
        p2_idx = int(p2_node.split("_")[1])
        sorted_ports1.append(ports1[p1_idx])
        sorted_ports2.append(ports2[p2_idx])

    return sorted_ports1, sorted_ports2


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

    # Find axis
    angle_start = ports1[0].orientation

    axis = "X" if angle_start in [0, 180] else "Y"
    # Sort ports using bipartite matching to avoid crossings
    ports1_sorted, ports2_sorted = _sort_ports_using_bipartite_matching(
        ports1, ports2, axis
    )

    # Ensure no crossing paths
    if enforce_port_ordering:
        ports2_sorted = [ports2_sorted[ports1_sorted.index(p1)] for p1 in ports1_sorted]

    return ports1_sorted, ports2_sorted
