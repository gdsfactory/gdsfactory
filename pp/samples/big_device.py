import numpy as np

import pp
from pp import LAYER, Port


@pp.cell
def big_device(w=400.0, h=400.0, N=16, port_pitch=15.0, layer=LAYER.WG, wg_width=0.5):
    """ big device with N ports on each side """
    component = pp.Component()
    p0 = np.array((0, 0))
    dx = w / 2
    dy = h / 2

    points = [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]]
    component.add_polygon(points, layer=layer)
    port_params = {"layer": layer, "width": wg_width}
    for i in range(N):
        port = Port(
            name=f"W{i}",
            midpoint=p0 + (-dx, (i - N / 2) * port_pitch),
            orientation=180,
            **port_params,
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name=f"E{i}",
            midpoint=p0 + (dx, (i - N / 2) * port_pitch),
            orientation=0,
            **port_params,
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name=f"N{i}",
            midpoint=p0 + ((i - N / 2) * port_pitch, dy),
            orientation=90,
            **port_params,
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name=f"S{i}",
            midpoint=p0 + ((i - N / 2) * port_pitch, -dy),
            orientation=-90,
            **port_params,
        )
        component.add_port(port)
    return component


if __name__ == "__main__":
    c = big_device()
    pp.show(c)
