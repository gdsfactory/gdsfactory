from __future__ import annotations

from typing import List, Tuple

from gdsfactory.port import Port


def get_port_x(port: Port) -> float:
    return port.center[0]


def get_port_y(port: Port) -> float:
    return port.center[1]


def sort_ports_x(ports: List[Port]) -> List[Port]:
    f_key = get_port_x
    ports.sort(key=f_key)
    return ports


def sort_ports_y(ports: List[Port]) -> List[Port]:
    f_key = get_port_y
    ports.sort(key=f_key)
    return ports


def sort_ports(ports1: List[Port], ports2: List[Port]) -> Tuple[List[Port], List[Port]]:
    """Returns two lists of sorted ports.

    Args:
        ports1:
        ports2:

    """
    if len(ports1) != len(ports2):
        raise ValueError(f"ports1={len(ports1)} and ports2={len(ports2)} must be equal")
    if not ports1:
        raise ValueError("ports1 is an empty list")
    if not ports2:
        raise ValueError("ports2 is an empty list")

    if isinstance(ports1, dict):
        ports1 = list(ports1.values())

    if isinstance(ports2, dict):
        ports2 = list(ports2.values())

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
        axis = "X" if ports1[0].orientation in [0, 180] else "Y"
        f_key1 = get_port_y if axis in {"X", "x"} else get_port_x
        ports2_by1 = dict(zip(ports1, ports2))
        ports1.sort(key=f_key1)
        ports2 = [ports2_by1[p1] for p1 in ports1]
    return ports1, ports2


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.cell import cell

    @cell
    def demo_connect_corner(N=6, config="A"):
        d = 10.0
        sep = 5.0
        top_cell = gf.Component(name="connect_corner")

        if config in ["A", "B"]:
            a = 100.0
            ports_A_TR = [
                Port(f"A_TR_{i}", (d, a / 2 + i * sep), 0.5, 0) for i in range(N)
            ]

            ports_A_TL = [
                Port(f"A_TL_{i}", (-d, a / 2 + i * sep), 0.5, 180) for i in range(N)
            ]

            ports_A_BR = [
                Port(f"A_BR_{i}", (d, -a / 2 - i * sep), 0.5, 0) for i in range(N)
            ]

            ports_A_BL = [
                Port(f"A_BL_{i}", (-d, -a / 2 - i * sep), 0.5, 180) for i in range(N)
            ]

            ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

            ports_B_TR = [
                Port(f"B_TR_{i}", (a / 2 + i * sep, d), 0.5, 90) for i in range(N)
            ]

            ports_B_TL = [
                Port(f"B_TL_{i}", (-a / 2 - i * sep, d), 0.5, 90) for i in range(N)
            ]

            ports_B_BR = [
                Port(f"B_BR_{i}", (a / 2 + i * sep, -d), 0.5, 270) for i in range(N)
            ]

            ports_B_BL = [
                Port(f"B_BL_{i}", (-a / 2 - i * sep, -d), 0.5, 270) for i in range(N)
            ]

            ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

        elif config in ["C", "D"]:
            a = N * sep + 2 * d
            ports_A_TR = [Port(f"A_TR_{i}", (a, d + i * sep), 0.5, 0) for i in range(N)]
            ports_A_TL = [
                Port(f"A_TL_{i}", (-a, d + i * sep), 0.5, 180) for i in range(N)
            ]
            ports_A_BR = [
                Port(f"A_BR_{i}", (a, -d - i * sep), 0.5, 0) for i in range(N)
            ]
            ports_A_BL = [
                Port(f"A_BL_{i}", (-a, -d - i * sep), 0.5, 180) for i in range(N)
            ]

            ports_A = [ports_A_TR, ports_A_TL, ports_A_BR, ports_A_BL]

            ports_B_TR = [
                Port(f"B_TR_{i}", (d + i * sep, a), 0.5, 90) for i in range(N)
            ]
            ports_B_TL = [
                Port(f"B_TL_{i}", (-d - i * sep, a), 0.5, 90) for i in range(N)
            ]
            ports_B_BR = [
                Port(f"B_BR_{i}", (d + i * sep, -a), 0.5, 270) for i in range(N)
            ]
            ports_B_BL = [
                Port(f"B_BL_{i}", (-d - i * sep, -a), 0.5, 270) for i in range(N)
            ]

            ports_B = [ports_B_TR, ports_B_TL, ports_B_BR, ports_B_BL]

        if config in ["A", "C"]:
            for ports1, ports2 in zip(ports_A, ports_B):
                routes = gf.routing.get_bundle(ports1, ports2, radius=8)
                for route in routes:
                    top_cell.add(route.references)

        elif config in ["B", "D"]:
            for ports1, ports2 in zip(ports_A, ports_B):
                routes = gf.routing.get_bundle(ports2, ports1, radius=8)
                for route in routes:
                    top_cell.add(route.references)

        return top_cell

    c = gf.Component()
    c1 = c << demo_connect_corner(config="A")
    c2 = c << demo_connect_corner(config="C")
    c2.xmin = c1.xmax + 5
    c.show(show_ports=True)
