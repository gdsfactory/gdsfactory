from typing import Dict, List, Tuple

import pp
from pp.component import Component

DIRECTION_TO_ANGLE = {"W": 180, "E": 0, "N": 90, "S": 270}


@pp.cell
def rectangle(
    size: Tuple[float, float] = (4.0, 2.0),
    layer: Tuple[int, int] = pp.LAYER.WG,
    centered: bool = False,
    ports: Dict[str, List[Tuple[float, float, float]]] = None,
    **port_settings
) -> Component:
    """rectangle

    Args:
        size: (tuple) Width and height of rectangle.
        layer: (int, array-like[2], or set) Specific layer(s) to put polygon geometry on.
        centered: True sets center to (0, 0)
        ports: {direction: [(x, y, width), ...]} direction: 'W', 'E', 'N' or 'S'

    """

    c = pp.Component()
    w, h = size

    if centered:
        points = [
            [-w / 2.0, -h / 2.0],
            [-w / 2.0, h / 2],
            [w / 2, h / 2],
            [w / 2, -h / 2.0],
        ]
    else:
        points = [[w, h], [w, 0], [0, 0], [0, h]]
    c.add_polygon(points, layer=layer)

    i = 0
    if ports:
        for direction, list_port_params in ports.items():
            assert direction in "NESW"
            angle = DIRECTION_TO_ANGLE[direction]
            for x, y, width in list_port_params:
                c.add_port(
                    name="{}".format(i),
                    orientation=angle,
                    midpoint=(x, y),
                    width=width,
                    layer=layer,
                    **port_settings
                )
                i += 1

    pp.port.auto_rename_ports(c)
    return c


if __name__ == "__main__":
    c = rectangle(size=(4, 2), ports={"N": [(0, 1, 4)]}, centered=True)
    print(c.ports)
    print(c.name)
    c.show()
