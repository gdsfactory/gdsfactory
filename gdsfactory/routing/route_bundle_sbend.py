from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.sort_ports import sort_ports as sort_ports_function
from gdsfactory.typings import ComponentSpec, Ports


def route_bundle_sbend(
    component: Component,
    ports1: Ports,
    ports2: Ports,
    bend_s: ComponentSpec = "bend_s",
    sort_ports: bool = True,
    enforce_port_ordering: bool = True,
    **kwargs: Any,
) -> None:
    """Places sbend routes from ports1 to ports2.

    Args:
        component: to place the sbend routes.
        ports1: start ports.
        ports2: end ports.
        bend_s: Sbend component.
        sort_ports: sort ports.
        enforce_port_ordering: enforces port ordering.
        kwargs: cross_section settings.

    """
    if sort_ports:
        ports1, ports2 = sort_ports_function(
            ports1, ports2, enforce_port_ordering=enforce_port_ordering
        )

    for p1, p2 in zip(list(ports1), list(ports2)):
        ysize = p2.center[1] - p1.center[1]
        xsize = p2.center[0] - p1.center[0]
        bend = gf.get_component(bend_s, size=(xsize, ysize), **kwargs)
        sbend = component << bend
        sbend.connect("o1", p1)


if __name__ == "__main__":
    import gdsfactory as gf

    # c = gf.Component(name="test_route_single_sbend")
    # pitch = 2.0
    # ys_left = [0, 10, 20]
    # N = len(ys_left)
    # y0 = -10
    # ys_right = [(i - N / 2) * pitch + y0 for i in range(N)]
    # layer = (1, 0)
    # right_ports = [
    #     gf.Port(
    #         f"R_{i}", center=(0, ys_right[i]), width=0.5, orientation=180, layer=layer
    #     )
    #     for i in range(N)
    # ]
    # left_ports = [
    #     gf.Port(
    #         f"L_{i}", center=(-50, ys_left[i]), width=0.5, orientation=0, layer=layer
    #     )
    #     for i in range(N)
    # ]
    # left_ports.reverse()
    # routes = gf.routing.route_bundle_sbend(
    #     c, right_ports, left_ports, enforce_port_ordering=False
    # )

    c = gf.Component()
    c1 = c << gf.components.nxn(east=3, ysize=20)
    c2 = c << gf.components.nxn(west=3)
    c2.dmove((80, 0))
    route_bundle_sbend(
        c,
        c1.ports.filter(orientation=0),
        c2.ports.filter(orientation=180),
        enforce_port_ordering=False,
    )
    c.show()
