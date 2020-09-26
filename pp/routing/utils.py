from typing import Dict, List
from pp.port import Port


def flip(port: Port) -> Port:
    """ Flip a phidl Port """
    return Port(port.name, port.midpoint, port.width, port.orientation + 180)


def direction_ports_from_list_ports(optical_ports: List[Port]) -> Dict[str, List[Port]]:
    direction_ports = {x: [] for x in ["E", "N", "W", "S"]}
    for p in optical_ports:
        p.angle = (p.angle + 360.0) % 360
        if p.angle <= 45.0 or p.angle >= 315:
            direction_ports["E"].append(p)
        elif p.angle <= 135.0 and p.angle >= 45.0:
            direction_ports["N"].append(p)
        elif p.angle <= 225.0 and p.angle >= 135.0:
            direction_ports["W"].append(p)
        else:
            direction_ports["S"].append(p)

    for direction, list_ports in list(direction_ports.items()):
        if direction in ["E", "W"]:
            list_ports.sort(key=lambda p: p.y)

        if direction in ["S", "N"]:
            list_ports.sort(key=lambda p: p.x)

    return direction_ports


def check_ports_have_equal_spacing(list_ports):
    if not list_ports:
        raise ValueError("list_ports should not be empty")

    angle = get_list_ports_angle(list_ports)
    if angle in [0, 180]:
        xys = [p.y for p in list_ports]
    else:
        xys = [p.x for p in list_ports]

    seps = [round(abs(c2 - c1), 5) for c1, c2 in zip(xys[1:], xys[:-1])]
    different_seps = set(seps)
    if len(different_seps) > 1:
        raise ValueError(
            "Ports should have the same separation. \
        Got {}".format(
                different_seps
            )
        )

    return different_seps.pop()


def get_list_ports_angle(list_ports):
    if not list_ports:
        return None
    if len(set([p.angle for p in list_ports])) > 1:
        raise ValueError(
            "All port angles should be the same.\
        Got {}".format(
                list_ports
            )
        )
    return list_ports[0].orientation
