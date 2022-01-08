"""

when we auto_rename_ports_layer_orientation we get the wrong mapping

"""

from typing import Dict
from gdsfactory.types import Port


def map_ports_layer_to_orientation(ports: Dict[str, Port]) -> Dict[str, str]:
    """Returns ports mapping

    {'1_0_W0': 1, '1_0_E0':2}

    .. code::

             N0  N1
             |___|_
        W1 -|      |- E1
            |      |
        W0 -|______|- E0
             |   |
            S0   S1

    """

    m = {}
    layers = {port.layer for port in ports.values()}

    for layer in layers:
        ports_on_layer = [p for p in ports.values() if p.layer == layer]
        direction_ports = {x: [] for x in ["E", "N", "W", "S"]}

        for p in ports_on_layer:
            angle = p.orientation % 360
            if angle <= 45 or angle >= 315:
                direction_ports["E"].append(p)
            elif angle <= 135 and angle >= 45:
                direction_ports["N"].append(p)
            elif angle <= 225 and angle >= 135:
                direction_ports["W"].append(p)
            else:
                direction_ports["S"].append(p)

        for direction, ports_sorted in direction_ports.items():
            if direction in ["N", "S"]:
                ports_sorted.sort(key=lambda p: +p.x)  # sort west to east
            else:
                ports_sorted.sort(key=lambda p: +p.y)  # sort south to north

            m.update(
                {
                    f"{layer[0]}_{layer[1]}_{direction}{i}": port.name
                    for i, port in enumerate(ports_sorted)
                }
            )

    return m


if __name__ == "__main__":
    from pprint import pprint
    import gdsfactory as gf

    # c = gf.Component()

    c = gf.components.straight_heater_metal()

    # FIXME, this line has an issue
    c.auto_rename_ports_layer_orientation()
    m = map_ports_layer_to_orientation(c.ports)
    pprint(m)
    c.show()
