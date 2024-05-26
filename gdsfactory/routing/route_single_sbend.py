from __future__ import annotations

from gdsfactory.components.bend_s import bend_s
from gdsfactory.port import Port
from gdsfactory.typings import Component


def route_single_sbend(
    component: Component, port1: Port, port2: Port, **kwargs
) -> None:
    """Returns an Sbend to connect two ports.

    Args:
        component: to add the route to.
        port1: start port.
        port2: end port.

    Keyword Args:
        npoints: number of points.
        with_cladding_box: square bounding box to avoid DRC errors.
        cross_section: function.
        kwargs: cross_section settings.

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.Component()
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.movex(50)
        mmi2.movey(5)
        route = gf.routing.route_single_sbend(c, mmi1.ports['o2'], mmi2.ports['o1'])
        c.plot()
    """
    ysize = port2.d.center[1] - port1.d.center[1]
    xsize = port2.d.center[0] - port1.d.center[0]

    # We need to act differently if the route is orthogonal in x
    # or orthogonal in y
    size = (xsize, ysize) if port1.orientation in [0, 180] else (ysize, -xsize)
    bend = bend_s(size=size, **kwargs)

    bend_ref = component << bend
    bend_ref.connect(bend_ref.ports[0], port1)

    orthogonality_error = abs(abs(port1.orientation - port2.orientation) - 180)
    if orthogonality_error > 0.1:
        raise ValueError(
            f"Ports need to have orthogonal orientation {orthogonality_error}\n"
            f"port1 = {port1.orientation} deg and port2 = {port2.orientation}"
        )


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("demo_route_sbend")
    mmi1 = c << gf.components.mmi1x2()
    mmi2 = c << gf.components.mmi1x2()
    mmi2.d.movex(50)
    mmi2.d.movey(5)
    route = route_single_sbend(c, mmi1.ports["o2"], mmi2.ports["o1"])
    c.show()
