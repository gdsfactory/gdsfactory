# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Klayout Design Rule Checking (DRC)
#
# Your device can be fabricated correctly when it meets the Design Rule Checks (DRC), you can write DRC rules from gdsfactory and customize the shortcut to run the checks in Klayout.
#
# Here are some rules explained in [repo generic DRC technology](https://github.com/klayoutmatthias/si4all) and [video](https://peertube.f-si.org/videos/watch/addc77a0-8ac7-4742-b7fb-7d24360ceb97)
#
# ![rules1](https://i.imgur.com/gNP5Npn.png)

# %%
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

# %%
help(write_drc_deck_macro)

# %%
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

# %% [markdown]
# # Klayout connectivity checks
#
# You can you can to check for component overlap and unconnected pins using klayout DRC.
#
#
# The easiest way is to write all the pins on the same layer and define the allowed pin widths.
# This will check for disconnected pins or ports with width mismatch.

# %%
from gdsfactory.generic_tech import LAYER
import gdsfactory.geometry.write_connectivity as wc

nm = 1e-3

rules = [
    wc.write_connectivity_checks(pin_widths=[0.5, 0.9, 0.45], pin_layer=LAYER.PORT)
]
script = wc.write_drc_deck_macro(rules=rules, layers=None)


# %% [markdown]
# You can also define the connectivity checks per section

# %%
connectivity_checks = [
    wc.ConnectivyCheck(cross_section="strip", pin_length=1 * nm, pin_layer=(1, 10)),
    wc.ConnectivyCheck(
        cross_section="strip_auto_widen", pin_length=1 * nm, pin_layer=(1, 10)
    ),
]
rules = [
    wc.write_connectivity_checks_per_section(connectivity_checks=connectivity_checks),
    "DEVREC",
]
script = wc.write_drc_deck_macro(rules=rules, layers=None)
