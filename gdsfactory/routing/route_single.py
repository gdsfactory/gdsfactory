"""`route_single` places a Manhattan route between two ports.

`route_single` only works for an individual routes. For routing groups of ports you need to use `route_bundle` instead

To make a route, you need to supply:

 - input port
 - output port
 - bend
 - straight
 - taper to taper to wider straights and reduce straight loss (Optional)

To generate a route:

 1. Generate the backbone of the route.
 This is a list of manhattan coordinates that the route would pass through
 if it used only sharp bends (right angles)

 2. Replace the corners by bend references
 (with rotation and position computed from the manhattan backbone)

 3. Add tapers if needed and if space permits

 4. generate straight portions in between tapers or bends

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import kfactory as kf
from kfactory.routing.generic import ManhattanRoute

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.routing.route_bundle import route_bundle
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    LayerSpec,
    LayerTransitions,
    Port,
    Step,
    WayPoints,
)


def route_single(
    component: Component,
    port1: Port,
    port2: Port,
    cross_section: CrossSectionSpec | None = None,
    layer: LayerSpec | None = None,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    start_straight_length: float = 0.0,
    end_straight_length: float = 0.0,
    waypoints: WayPoints | None = None,
    steps: Sequence[Step] | None = None,
    port_type: str | None = None,
    allow_width_mismatch: bool = False,
    radius: float | None = None,
    route_width: float | None = None,
    auto_taper: bool = True,
    on_error: Literal["error"] | None = None,
    layer_transitions: LayerTransitions | None = None,
) -> ManhattanRoute:
    """Returns a Manhattan Route between 2 ports.

    The references are straights, bends and tapers.

    Args:
        component: to place the route into.
        port1: start port.
        port2: end port.
        cross_section: spec.
        layer: layer spec.
        bend: bend spec.
        straight: straight spec.
        start_straight_length: length of starting straight.
        end_straight_length: length of end straight.
        waypoints: optional list of points to pass through.
        steps: optional list of steps to pass through.
            Each step is a dict with keys: x (absolute), y (absolute), dx (relative), dy (relative).
            Use x/y to set an absolute coordinate and dx/dy to shift relative to the current position.
        port_type: port type to route.
        allow_width_mismatch: allow different port widths.
        radius: bend radius. If None, defaults to cross_section.radius.
        route_width: width of the route in um. If None, defaults to cross_section.width.
        auto_taper: add auto tapers.
        on_error: what to do on error. If error, raises an error. If None ignores the error.
        layer_transitions: dictionary of layer transitions to use for the routing when auto_taper=True.

    Example:
        ```python
        import gdsfactory as gf

        c = gf.Component()
        mmi1 = c << gf.components.mmi1x2()
        mmi2 = c << gf.components.mmi1x2()
        mmi2.move((40, 20))
        gf.routing.route_single(c, mmi1.ports["o2"], mmi2.ports["o1"], radius=5, cross_section="strip")
        c.plot()
        ```
    """
    return route_bundle(
        component=component,
        ports1=[port1],
        ports2=[port2],
        cross_section=cross_section,
        layer=layer,
        bend=bend,
        straight=straight,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        waypoints=waypoints,
        steps=steps,
        port_type=port_type,
        allow_width_mismatch=allow_width_mismatch,
        radius=radius,
        route_width=route_width,
        auto_taper=auto_taper,
        raise_on_error=on_error == "error",
        layer_transitions=layer_transitions,
    )[0]


def route_single_electrical(
    component: Component,
    port1: Port,
    port2: Port,
    start_straight_length: float | None = None,
    end_straight_length: float | None = None,
    layer: LayerSpec | None = None,
    width: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> None:
    """Places a route between two electrical ports.

    Args:
        component: The cell to place the route in.
        port1: The first port.
        port2: The second port.
        start_straight_length: The length of the straight at the start of the route.
        end_straight_length: The length of the straight at the end of the route.
        layer: The layer of the route.
        width: The width of the route.
        cross_section: The cross section of the route.

    """
    xs = gf.get_cross_section(cross_section)
    layer = layer or xs.layer
    width = width or xs.width
    layer = gf.get_layer(layer)
    kf.routing.electrical.route_bundle(
        c=component,
        start_ports=[port1],
        end_ports=[port2],
        separation=0,
        route_width=width,
        place_layer=kf.kdb.LayerInfo(layer[0], layer[1])
        if isinstance(layer, tuple)
        else None,
        starts=start_straight_length,
        ends=end_straight_length,
    )
