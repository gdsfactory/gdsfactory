from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory import Port
from gdsfactory.components.tapers import taper
from gdsfactory.difftest import difftest
from gdsfactory.gpdk import LAYER
from gdsfactory.routing.route_bundle import route_bundle


def test_route_bundle_auto_taper_layer_transitions() -> None:
    """Tests route_bundle with auto_taper and layer transitions."""
    layer = LAYER.WG
    transition = partial(taper, length=13.5)
    layer_transitions = {LAYER.WG: transition}
    pitch = 127.0
    xs_top = [0, 50, 100]
    N = len(xs_top)
    xs_bottom = [(i - N / 2) * pitch for i in range(N)]
    layer = 1

    top_ports = [
        Port(
            name=f"top_{i}",
            center=(xs_top[i], 0),
            width=2.7 - i * 0.2,
            orientation=270,
            layer=layer,
        )
        for i in range(N)
    ]

    bot_ports = [
        Port(
            name=f"bot_{i}",
            center=(xs_bottom[i], -400),
            width=2.1 - i * 0.1,
            orientation=90,
            layer=layer,
        )
        for i in range(N)
    ]

    c = gf.Component(name="test_route_bundle_auto_taper_layer_transitions")
    route_bundle(
        c,
        top_ports,
        bot_ports,
        start_straight_length=5,
        end_straight_length=10,
        cross_section="strip",
        auto_taper=True,
        layer_transitions=layer_transitions,
    )
    # lengths = {i: route.length for i, route in enumerate(routes)}
    difftest(c)


def test_route_bundle_length_includes_auto_tapers() -> None:
    transition = partial(taper, length=10)
    layer_index = gf.get_layer(LAYER.WG)
    port1 = Port(name="o1", center=(0, 0), width=1, orientation=0, layer=layer_index)
    port2 = Port(
        name="o2", center=(100, 0), width=1, orientation=180, layer=layer_index
    )
    component = gf.Component()

    route = route_bundle(
        component,
        [port1],
        [port2],
        cross_section="strip",
        auto_taper=True,
        layer_transitions={LAYER.WG: transition},
    )[0]

    assert route.length == 100_000
