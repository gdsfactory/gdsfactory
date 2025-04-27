from __future__ import annotations

from typing import Any

from kfactory.routing.generic import ManhattanRoute

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
    allow_width_mismatch: bool | None = None,
    allow_layer_mismatch: bool | None = None,
    allow_type_mismatch: bool | None = None,
    port_name: str = "o1",
    use_port_width: bool = True,
    **kwargs: Any,
) -> list[ManhattanRoute]:
    """Places sbend routes from ports1 to ports2.

    Args:
        component: to place the sbend routes.
        ports1: start ports.
        ports2: end ports.
        bend_s: Sbend component.
        sort_ports: sort ports.
        enforce_port_ordering: enforces port ordering.
        allow_width_mismatch: allows width mismatch.
        allow_layer_mismatch: allows layer mismatch.
        allow_type_mismatch: allows type mismatch.
        port_name: name of the port to connect to the sbend.
        use_port_width: if True, use the width of the port to set the width of the sbend.
        kwargs: cross_section settings.

    """
    if sort_ports:
        ports1, ports2 = sort_ports_function(
            ports1, ports2, enforce_port_ordering=enforce_port_ordering
        )

    routes = []

    for p1, p2 in zip(list(ports1), list(ports2)):
        ys = p2.center[1] - p1.center[1]
        xs = p2.center[0] - p1.center[0]

        if p1.orientation in [0, 180]:
            xsize = xs
            ysize = ys
        elif p1.orientation == 90:
            xsize = ys
            ysize = -xs

        elif p1.orientation == 270:
            xsize = -ys
            ysize = xs

        if use_port_width:
            bend = gf.get_component(
                bend_s, size=(xsize, ysize), width=p1.width, **kwargs
            )
        else:
            bend = gf.get_component(bend_s, size=(xsize, ysize), **kwargs)
        sbend = component << bend
        sbend.connect(
            port_name,
            p1,
            allow_width_mismatch=allow_width_mismatch,
            allow_layer_mismatch=allow_layer_mismatch,
            allow_type_mismatch=allow_type_mismatch,
        )

        route = ManhattanRoute(
            backbone=[],
            start_port=p1.to_itype(),
            end_port=p2.to_itype(),
            instances=[],
            bend90_radius=round(bend.info.get("min_bend_radius", 0)),
        )
        routes.append(route)
    return routes


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

    # c = gf.Component()
    # c1 = c << gf.components.nxn(east=3, ysize=20)
    # c2 = c << gf.components.nxn(west=3)
    # c2.move((80, 0))
    # route_bundle_sbend(
    #     c,
    #     c1.ports.filter(orientation=0),
    #     c2.ports.filter(orientation=180),
    #     enforce_port_ordering=False,
    # )

    c = gf.Component()
    c1 = c << gf.components.straight_heater_metal()
    p1 = c << gf.c.pad()
    p1.movey(200)
    route_bundle_sbend(
        c,
        [p1["e4"]],
        [c1.ports["l_e2"]],
        enforce_port_ordering=False,
        cross_section="metal3",
        port_name="e1",
        allow_width_mismatch=True,
    )
    c.show()
