from __future__ import annotations

import functools
from collections.abc import Sequence

from gdsfactory.component import Component
from gdsfactory.routing.route_bundle import route_bundle, route_bundle_electrical
from gdsfactory.routing.route_bundle_all_angle import route_bundle_all_angle
from gdsfactory.routing.route_bundle_sbend import route_bundle_sbend
from gdsfactory.typings import Port, Ports, Route, RoutingStrategies, RoutingStrategy


def support_nets(func: RoutingStrategy) -> RoutingStrategy:
    """Adapt a ``ports1``/``ports2`` bundle router to also accept schematic nets.

    kfactory's schematic ``create_cell`` calls routing strategies as
    ``strategy(component, nets, **settings)`` where ``nets`` is a sequence of
    port tuples (one net per pair of ports to connect). gdsfactory's
    ``from_yaml`` instead calls them as
    ``strategy(component, ports1=..., ports2=..., **settings)``.

    This wrapper accepts both conventions so the same strategy can be used from
    a YAML netlist and from a python ``Schematic``.
    """

    @functools.wraps(func)
    def wrapper(
        component: Component,
        nets: Sequence[Sequence[Port]] | None = None,
        *,
        ports1: Ports | None = None,
        ports2: Ports | None = None,
        **kwargs: object,
    ) -> Sequence[Route]:
        if nets is not None:
            ports1 = [net[0] for net in nets]
            ports2 = [net[1] for net in nets]
        return func(component, ports1=ports1, ports2=ports2, **kwargs)

    return wrapper


routing_strategies: RoutingStrategies = {
    "route_bundle": support_nets(route_bundle),
    "route_bundle_electrical": support_nets(route_bundle_electrical),
    "route_bundle_all_angle": support_nets(route_bundle_all_angle),
    "route_bundle_sbend": support_nets(route_bundle_sbend),
}
