from collections.abc import Sequence
from functools import partial

from kfactory.routing.aa.optical import (
    BendFactory,
    StraightFactory,
    route_bundle,
)

from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, Port


def route_bundle_all_angle(
    component: ComponentSpec,
    ports1: list[Port],
    ports2: list[Port],
    backbone: Sequence[tuple[float, float]],
    separation: list[float] | float,
    straight_factory: StraightFactory = straight,
    bend_factory: BendFactory = partial(bend_euler, radius=5),
    bend_ports: tuple[str, str] = ("o1", "o2"),
    straight_ports: tuple[str, str] = ("o1", "o2"),
):
    """Route a bundle of ports to another bundle of ports with all angles.

    Args:
        component: to add the routing.
        ports1: list of start ports to connect.
        ports2: list of end ports to connect.
        backbone: list of points to connect the ports.
        separation: list of spacings.
        straight_factory: factory to create straights.
        bend_factory: factory to create bends.
        bend_ports: tuple of ports to connect the bends.
        straight_ports: tuple of ports to connect the straights.
    """

    backbone = [gf.kdb.DPoint(*p) for p in backbone]

    return route_bundle(
        c=component,
        start_ports=ports1,
        end_ports=ports2,
        backbone=backbone,
        separation=separation,
        straight_factory=straight_factory,
        bend_factory=bend_factory,
        bend_ports=bend_ports,
        straight_ports=straight_ports,
    )


if __name__ == "__main__":
    import numpy as np

    import gdsfactory as gf

    c = gf.Component()
    rows = 3
    w1 = c << gf.c.array("straight", spacing=(0, 10), rows=rows, columns=1)
    w2 = c << gf.c.array("straight", spacing=(0, 10), rows=rows, columns=1)
    w2.d.rotate(-30)
    w2.d.movex(140)
    p1 = list(w1.ports.filter(orientation=0))
    p2 = list(w2.ports.filter(orientation=150))
    p1.reverse()
    p2.reverse()

    c1 = np.array(p2[0].d.center)
    c2 = np.array(p1[0].d.center)
    d = (np.array(p2[0].d.center) + np.array(p1[0].d.center)) / 2
    backbone = [
        d - (10.0, 0),
        d + (10.0, 0),
    ]

    route_bundle_all_angle(
        c,
        p1,
        p2,
        backbone=backbone,
        separation=3,
    )
    c.show()
