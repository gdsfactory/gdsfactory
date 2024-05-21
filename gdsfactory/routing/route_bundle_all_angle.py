from collections.abc import Sequence

from kfactory import kdb
from kfactory.routing.aa.optical import (
    BendFactory,
    RadiusEstimate,
    StraightFactory,
    route_bundle,
)

from gdsfactory.typings import ComponentSpec, Port


def route_bundle_all_angle(
    component: ComponentSpec,
    ports1: list[Port],
    ports2: list[Port],
    backbone: Sequence[kdb.DPoint],
    spacings: list[float],
    radius_estimate: RadiusEstimate,
    straight_factory: StraightFactory,
    bend_factory: BendFactory,
    bend_ports: tuple[str, str] = ("o1", "o2"),
    straight_ports: tuple[str, str] = ("o1", "o2"),
):
    """Route a bundle of ports to another bundle of ports with all angles.

    Args:
        component: to add the routing
        ports1: list of start ports to connect
        ports2: list of end ports to connect
        backbone: list of points to connect the ports
        spacings: list of spacings
        radius_estimate: estimate the radius
        straight_factory: factory to create straights
        bend_factory: factory to create bends
        bend_ports: tuple of ports to connect the bends
        straight_ports: tuple of ports to connect the straights
    """
    return route_bundle(
        component=component,
        start_ports=ports1,
        end_ports=ports2,
        backbone=backbone,
        spacings=spacings,
        radius_estimate=radius_estimate,
        straight_factory=straight_factory,
        bend_factory=bend_factory,
        bend_ports=bend_ports,
        straight_ports=straight_ports,
    )


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component()
    w1 = c << gf.c.array("straight", spacing=(0, 10), rows=3, columns=1)
    w2 = c << gf.c.array("straight", spacing=(0, 10), rows=3, columns=1)
    w2.d.rotate(-30)
    w2.d.movex(100)

    route_bundle_all_angle(
        c,
        w1.ports.filter(orientation=0),
        w2.ports.filter(orientation=150),
    )
    c.show()
