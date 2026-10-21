from collections.abc import Callable

import pytest

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components import straight, taper, taper_sc_nc
from gdsfactory.cross_section import strip
from gdsfactory.gpdk.layer_map import LAYER
from gdsfactory.port import Port
from gdsfactory.routing.auto_taper import add_auto_tapers, auto_taper_to_cross_section
from gdsfactory.typings import LayerTransitions

LAYER_TRANSITIONS: LayerTransitions = {(LAYER.WG, LAYER.WGN): taper_sc_nc}


def test_auto_taper() -> None:
    c = Component()
    ref = c << straight(cross_section="strip")
    auto_taper_to_cross_section(
        c,
        port=ref.ports["o2"],
        cross_section="nitride",
        layer_transitions=LAYER_TRANSITIONS,
    )


def test_auto_taper_reversed() -> None:
    c = Component()
    ref = c << straight(cross_section="nitride")
    auto_taper_to_cross_section(
        c,
        port=ref.ports["o2"],
        cross_section="strip",
        layer_transitions=LAYER_TRANSITIONS,
    )


def test_auto_taper_layer_transitions() -> None:
    c = Component()
    ref = c << straight(cross_section="strip")
    auto_taper_to_cross_section(
        c,
        port=ref.ports["o2"],
        cross_section=strip(width=0.75),
        layer_transitions={"WG": taper},
    )


@pytest.mark.parametrize(
    "route",
    [
        lambda c, p1, p2: add_auto_tapers(c, [p1, p2], cross_section="strip"),
        lambda c, p1, p2: gf.routing.route_single(c, p1, p2, cross_section="strip"),
        lambda c, p1, p2: gf.routing.route_bundle(c, [p1], [p2], cross_section="strip"),
        lambda c, p1, p2: gf.routing.route_bundle_sbend(
            c, [p1], [p2], cross_section="strip"
        ),
    ],
    ids=["add_auto_tapers", "route_single", "route_bundle", "route_bundle_sbend"],
)
def test_untaperable_port_leaves_component_unchanged(
    route: Callable[[Component, Port, Port], object],
) -> None:
    c = Component()
    start = c << straight(cross_section="strip", width=0.8)
    end = c << straight(
        cross_section=gf.cross_section.cross_section(layer="SLAB150", width=0.8)
    )
    end.move((300, 20))
    with pytest.raises(KeyError, match="No registered tapers"):
        route(c, start.ports[1], end.ports[0])
    # No auto-taper was left dangling on the start side.
    assert len(c.insts) == 2
