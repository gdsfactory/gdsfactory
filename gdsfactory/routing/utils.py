from __future__ import annotations

from typing import Dict, List, Union

from numpy import float64

from gdsfactory.port import Port


def flip(port: Port) -> Port:
    """Returns port copy with Flip Port orientation."""
    return port.flip()


def direction_ports_from_list_ports(optical_ports: List[Port]) -> Dict[str, List[Port]]:
    """Returns a dict of WENS ports."""
    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}
    for p in optical_ports:
        p.orientation = (p.orientation + 360.0) % 360
        if p.orientation <= 45.0 or p.orientation >= 315:
            direction_ports["E"].append(p)
        elif p.orientation <= 135.0 and p.orientation >= 45.0:
            direction_ports["N"].append(p)
        elif p.orientation <= 225.0 and p.orientation >= 135.0:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    for direction, list_ports in list(direction_ports.items()):
        if direction in ["E", "W"]:
            list_ports.sort(key=lambda p: p.y)

        if direction in ["S", "N"]:
            list_ports.sort(key=lambda p: p.x)

    return direction_ports


def check_ports_have_equal_spacing(list_ports: List[Port]) -> float64:
    """Returns port separation.

    Raises error if not constant.

    """
    if not isinstance(list_ports, list):
        raise ValueError(f"list_ports should be a list of ports, got {list_ports}")
    if not list_ports:
        raise ValueError("list_ports should not be empty")

    orientation = get_list_ports_angle(list_ports)
    if orientation in [0, 180]:
        xys = [p.y for p in list_ports]
    else:
        xys = [p.x for p in list_ports]

    seps = [round(abs(c2 - c1), 5) for c1, c2 in zip(xys[1:], xys[:-1])]
    different_seps = set(seps)
    if len(different_seps) > 1:
        raise ValueError("Ports should have the same separation. Got {different_seps}")

    return different_seps.pop()


def get_list_ports_angle(list_ports: List[Port]) -> Union[float64, int]:
    """Returns the orientation/angle (in degrees) of a list of ports."""
    if not list_ports:
        return None
    if len({p.orientation for p in list_ports}) > 1:
        raise ValueError(f"All port angles should be the same. Got {list_ports}")
    return list_ports[0].orientation


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.components.mmi1x2()
    d = direction_ports_from_list_ports(c.get_ports_list())
    c.show(show_ports=True)
