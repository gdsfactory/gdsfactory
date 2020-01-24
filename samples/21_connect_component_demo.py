"""
Connecting a component with I/O
"""

import numpy as np
import pp
from pp.layers import LAYER
from pp import Port
from pp.routing.connect_component import add_io_optical


@pp.autoname
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
            name="W{}".format(i),
            midpoint=p0 + (-dx, (i - N / 2) * port_pitch),
            orientation=180,
            **port_params
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name="E{}".format(i),
            midpoint=p0 + (dx, (i - N / 2) * port_pitch),
            orientation=0,
            **port_params
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name="N{}".format(i),
            midpoint=p0 + ((i - N / 2) * port_pitch, dy),
            orientation=90,
            **port_params
        )
        component.add_port(port)

    for i in range(N):
        port = Port(
            name="S{}".format(i),
            midpoint=p0 + ((i - N / 2) * port_pitch, -dy),
            orientation=-90,
            **port_params
        )
        component.add_port(port)
    return component


if __name__ == "__main__":
    component = big_device(N=10)
    bend_radius = 5.0

    # import cProfile, pstats, io

    # pr = cProfile.Profile()

    # pr.enable()
    # Had to fix the fanout length by hand here to give enough space.

    c = add_io_optical(component, bend_radius=bend_radius, fanout_length=50.0)
    # pr.disable()

    # s = io.StringIO()
    # sortby = "cumtime"
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    pp.show(c)
