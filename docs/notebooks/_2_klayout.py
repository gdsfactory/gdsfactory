# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # KLayout integration
#
# [Klayout](https://www.klayout.de/) is an open source layout viewer and editor. In gdsfactory code driven flow you define your components, circuits and reticles in python or YAML code.
#
# For rapid iteration, gdsfactory comes with a Klayout macro `klive` that runs inside klayout, so that when you run `component.show()` in python, it displays your GDS in Klayout.
#
# ![workflow](https://i.imgur.com/yquTcM7.png)

# You can install the klayout klive plugin to be able to see live updates on your GDS files:

# ![KLayout package](https://i.imgur.com/IZWH6U0.png)

# You can install the klayout integration to install the generic layermap:
#
# - from the terminal by typing `gf install klayout-integration` after installing gdsfactory `pip install gdsfactory`
# - using KLayout package manager (see image below), Tools --> Manage Packages
#
# ![KLayout package](https://i.imgur.com/AkfcCms.png)

# What does the klayout integration installs?
#
# - generic layermap: for the generic gdsfactory technology.
# - generic DRC: for generic gdsfactory technology

# ## KLayout Design Rule Checking (DRC)
#
# Your device can be fabricated correctly when it meets the Design Rules, you can write DRC rules from gdsfactory and customize the shortcut to run the checks in Klayout.
#
# Here are some rules explained in [repo generic DRC technology](https://github.com/klayoutmatthias/si4all) and [video](https://peertube.f-si.org/videos/watch/addc77a0-8ac7-4742-b7fb-7d24360ceb97)
#
# ![rules1](https://i.imgur.com/gNP5Npn.png)

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

help(write_drc_deck_macro)

# +
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
# -

# ## KLayout connectivity checks
#
# Thanks to [SiEPIC-Tools](https://github.com/SiEPIC/SiEPIC-Tools) klayout macro gdsfactory you can run component overlap and connectivity checks.
#
# This is enabled by default for any components in the [ubcpdk](https://gdsfactory.github.io/ubc/README.html) cross_section, thanks to having `add_pins=add_pins_siepic` and `add_bbox=add_bbox_siepic` by default.
#
# ![Siepic](https://i.imgur.com/wHnWxMb.png)
