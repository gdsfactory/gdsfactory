from gdsfactory.component import Component
from gdsfactory.components import straight, taper_sc_nc
from gdsfactory.generic_tech.layer_map import LAYER
from gdsfactory.routing.auto_taper import auto_taper_to_cross_section

LAYER_TRANSITIONS = {(LAYER.WG, LAYER.WGN): taper_sc_nc}


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
