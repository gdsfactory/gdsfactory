from __future__ import annotations

from kfactory.routing.aa.optical import OpticalAllAngleRoute, route_bundle

import gdsfactory as gf
from gdsfactory.typings import (
    CellAllAngleSpec,
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    Port,
)
from gdsfactory.utils import to_kdb_dpoints


def route_bundle_all_angle(
    component: ComponentSpec,
    ports1: list[Port],
    ports2: list[Port],
    backbone: Coordinates | None = None,
    separation: list[float] | float = 3.0,
    straight: CellAllAngleSpec = "straight_all_angle",
    bend: CellAllAngleSpec = "bend_euler_all_angle",
    bend_ports: tuple[str, str] = ("o1", "o2"),
    straight_ports: tuple[str, str] = ("o1", "o2"),
    cross_section: CrossSectionSpec | None = None,
) -> list[OpticalAllAngleRoute]:
    """Route a bundle of ports to another bundle of ports with non manhattan ports.

    Args:
        component: to add the routing.
        ports1: list of start ports to connect.
        ports2: list of end ports to connect.
        backbone: list of points to connect the ports.
        separation: list of spacings.
        straight: function to create straights.
        bend: function to create bends.
        bend_ports: tuple of ports to connect the bends.
        straight_ports: tuple of ports to connect the straights.
        cross_section: cross_section to use. Overrides the  cross_section.
    """
    if cross_section:
        straight_func = gf.get_cell(straight, cross_section=cross_section)
        bend_func = gf.get_cell(bend, cross_section=cross_section)

    else:
        straight_func = gf.get_cell(straight)
        bend_func = gf.get_cell(bend)

    backbone = backbone or []

    c = gf.get_component(component)

    return route_bundle(
        c=c,
        start_ports=ports1,
        end_ports=ports2,
        backbone=to_kdb_dpoints(backbone),
        separation=separation,
        straight_factory=straight_func,
        bend_factory=bend_func,
        bend_ports=bend_ports,
        straight_ports=straight_ports,
    )
