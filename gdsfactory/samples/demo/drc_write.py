"""Sample DRC deck for the generic PDK."""
from __future__ import annotations


import gdsfactory as gf
from gdsfactory.geometry.write_drc import (
    rule_area,
    rule_density,
    rule_enclosing,
    rule_separation,
    rule_space,
    rule_width,
    write_drc_deck_macro,
)

if __name__ == "__main__":
    rules = [
        rule_width(layer="WG", value=0.2),
        rule_space(layer="WG", value=0.2),
        rule_width(layer="M1", value=1),
        rule_width(layer="M2", value=2),
        rule_space(layer="M2", value=2),
        rule_separation(layer1="HEATER", layer2="M1", value=1.0),
        rule_enclosing(layer1="M1", layer2="VIAC", value=0.2),
        rule_area(layer="WG", min_area_um2=0.05),
        rule_density(
            layer="WG", layer_floorplan="FLOORPLAN", min_density=0.5, max_density=0.6
        ),
    ]

    drc_rule_deck = write_drc_deck_macro(
        rules=rules,
        layers=gf.LAYER,
        shortcut="Ctrl+Shift+D",
    )
    print(drc_rule_deck)
