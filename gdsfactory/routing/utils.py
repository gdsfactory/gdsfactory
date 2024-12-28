from __future__ import annotations

from collections.abc import Sequence

from gdsfactory.typings import Port


class RouteWarning(UserWarning):
    pass


def direction_ports_from_list_ports(
    optical_ports: Sequence[Port],
) -> dict[str, list[Port]]:
    """Returns a dict of WENS ports."""
    direction_ports: dict[str, list[Port]] = {x: [] for x in ["E", "N", "W", "S"]}
    for p in optical_ports:
        orientation = (p.orientation + 360.0) % 360
        if orientation <= 45.0 or orientation >= 315:
            direction_ports["E"].append(p)
        elif orientation <= 135.0 and orientation >= 45.0:
            direction_ports["N"].append(p)
        elif orientation <= 225.0 and orientation >= 135.0:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    for direction, list_ports in list(direction_ports.items()):
        if direction in ["E", "W"]:
            list_ports.sort(key=lambda p: p.dy)

        if direction in ["S", "N"]:
            list_ports.sort(key=lambda p: p.dx)

    return direction_ports


def check_ports_have_equal_spacing(list_ports: Sequence[Port]) -> float:
    """Returns port separation.

    Raises error if not constant.

    """
    if not isinstance(list_ports, list):
        raise ValueError(f"list_ports should be a list of ports, got {list_ports}")
    if not list_ports:
        raise ValueError("list_ports should not be empty")

    orientation = get_list_ports_angle(list_ports)
    if orientation in [0, 180]:
        xys = [p.dy for p in list_ports]
    else:
        xys = [p.dx for p in list_ports]

    seps = [round(abs(c2 - c1), 5) for c1, c2 in zip(xys[1:], xys[:-1])]
    different_seps = set(seps)
    if len(different_seps) > 1:
        raise ValueError(f"Ports should have the same separation. Got {different_seps}")
    return float(seps[0])


def get_list_ports_angle(list_ports: Sequence[Port]) -> float | None:
    """Returns the orientation/angle (in degrees) of a list of ports."""
    if not list_ports:
        return None
    if len({p.orientation for p in list_ports}) > 1:
        raise ValueError(f"All port angles should be the same. Got {list_ports}")
    return list_ports[0].orientation


if __name__ == "__main__":
    from gdsfactory.components import mmi1x2

    c = mmi1x2()
    d = direction_ports_from_list_ports(c.get_ports_list())
    c.show()
