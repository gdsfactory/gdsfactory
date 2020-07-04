import pp
from pp.component import Component
from typing import Dict, List, Tuple

DIRECTION_TO_ANGLE = {"W": 180, "E": 0, "N": 90, "S": 270}


@pp.autoname
def rectangle(
    size: Tuple[float, float] = (4.0, 2.0),
    layer: Tuple[int, int] = pp.LAYER.WG,
    centered: bool = False,
    ports_parameters: Dict[str, List[Tuple[float, float]]] = {},
    **port_settings
) -> Component:
    """ rectangle

    Args:
        size: (tuple) Width and height of rectangle.
        layer: (int, array-like[2], or set) Specific layer(s) to put polygon geometry on.
        ports: {direction: [(x_or_y, width), ...]} direction: 'W', 'E', 'N' or 'S'

    .. plot::
      :include-source:

      import pp

      c = pp.c.rectangle(size=(4, 2), layer=0)
      pp.plotgds(c)
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
    for direction, list_port_params in ports_parameters.items():
        angle = DIRECTION_TO_ANGLE[direction]
        for x_or_y, width in list_port_params:
            if direction == "W":
                position = (0, x_or_y)

            elif direction == "E":
                position = (w, x_or_y)

            elif direction == "S":
                position = (x_or_y, 0)

            elif direction == "N":
                position = (x_or_y, h)

            c.add_port(
                name="{}".format(i),
                orientation=angle,
                midpoint=position,
                width=width,
                layer=layer,
                **port_settings
            )
            i += 1

    pp.port.auto_rename_ports(c)
    return c


@pp.autoname
def rectangle_centered(
    w: int = 1, h: int = 1, x: None = None, y: None = None, layer: int = 0
) -> Component:
    """ a rectangle size (x, y) in layer
        bad naming with x and y. Replaced with w and h. Keeping x and y
        for now for backwards compatibility

    .. plot::
      :include-source:

      import pp

      c = pp.c.rectangle_centered(w=1, h=1, layer=0)
      pp.plotgds(c)
    """
    c = pp.Component()
    if x:
        w = x
    if y:
        h = y

    points = [
        [-w / 2.0, -h / 2.0],
        [-w / 2.0, h / 2],
        [w / 2, h / 2],
        [w / 2, -h / 2.0],
    ]
    c.add_polygon(points, layer=layer)
    return c


if __name__ == "__main__":
    # c = rectangle_centered()
    c = rectangle()
    print(c.ports)
    pp.show(c)
