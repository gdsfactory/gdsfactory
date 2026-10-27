from collections.abc import Callable
from functools import partial

import pytest

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components import straight, taper, taper_sc_nc
from gdsfactory.cross_section import strip
from gdsfactory.gpdk.layer_map import LAYER
from gdsfactory.port import Port
from gdsfactory.routing.auto_taper import add_auto_tapers, auto_taper_to_cross_section
from gdsfactory.typings import ComponentSpec, LayerTransitions

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


def _port(name: str, cross_section: gf.CrossSection) -> Port:
    return Port(name=name, center=(0, 0), cross_section=cross_section, orientation=0)


def test_add_auto_tapers_keeps_port_order() -> None:
    c = Component()
    ports = [
        _port("matching", strip()),
        _port("wide", strip(width=0.8)),
        _port("nitride", gf.cross_section.nitride()),  # reversed transition
    ]
    result = add_auto_tapers(
        c,
        ports,
        cross_section="strip",
        layer_transitions={LAYER.WG: taper, **LAYER_TRANSITIONS},
    )
    assert [p.name for p in result] == ["matching", "wide", "nitride"]
    assert result[0] is ports[0]
    assert all(p.width == 0.5 for p in result)
    assert len(c.insts) == 2


def test_add_auto_tapers_skips_missing_width_taper() -> None:
    c = Component()
    port = _port("o1", strip(width=0.8))
    with pytest.warns(UserWarning, match="No registered width taper"):
        result = add_auto_tapers(c, [port], cross_section="strip", layer_transitions={})
    assert result == [port]
    assert len(c.insts) == 0


@pytest.mark.parametrize(
    ("width_taper", "match"),
    [
        (partial(taper, cross_section="nitride"), "do not match"),
        ("taper_electrical", "should have two ports"),
    ],
)
def test_invalid_taper_leaves_component_unchanged(
    width_taper: ComponentSpec, match: str
) -> None:
    c = Component()
    # The nitride port is taperable, the wide one is not.
    ports = [
        _port("nitride", gf.cross_section.nitride()),
        _port("wide", strip(width=0.8)),
    ]
    with pytest.raises(ValueError, match=match):
        add_auto_tapers(
            c,
            ports,
            cross_section="strip",
            layer_transitions={LAYER.WG: width_taper, **LAYER_TRANSITIONS},
        )
    assert len(c.insts) == 0
