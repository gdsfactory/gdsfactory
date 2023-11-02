"""You can import gdsfactory.as gf.

functions:
    - import_gds(): returns a Component from a GDS

classes:

    - Component
    - Port
    - TECH

modules:

    - c: components
    - routing
"""
# isort: skip_file
from __future__ import annotations
from functools import partial
from toolz import compose
from gdsfactory.path import Path
import kfactory as kf
from kfactory.kcell import LayerEnum, kcl, show
import klayout.db as kdb

# NOTE: import order matters. Only change the order if you know what you are doing
from gdsfactory.component import Component, ComponentReference, Instance
from gdsfactory.config import CONF, call_if_func, PATH, logger
from gdsfactory.port import Port
from gdsfactory.cell import cell, cell_without_validator
from gdsfactory.read.import_gds import import_gds
from gdsfactory.cross_section import CrossSection, Section
from gdsfactory.difftest import difftest, diff
from gdsfactory.boolean import boolean

from gdsfactory import cross_section
from gdsfactory import asserts
from gdsfactory import port
from gdsfactory import components
from gdsfactory import typings
from gdsfactory import path
from gdsfactory import snap
from gdsfactory import read
from gdsfactory import add_ports
from gdsfactory import write_cells
from gdsfactory import add_pins
from gdsfactory import technology

from gdsfactory.add_padding import (
    add_padding,
    add_padding_container,
    get_padding_points,
)
from gdsfactory.pack import pack
from gdsfactory.grid import grid, grid_with_text
from gdsfactory.pdk import (
    Pdk,
    get_component,
    get_cross_section,
    get_layer,
    get_active_pdk,
    get_cell,
    get_constant,
)
from gdsfactory.get_factories import get_cells
from gdsfactory.cross_section import get_cross_sections
from aenum import constant  # type: ignore[import-untyped]

c = components


__all__ = (
    "CONF",
    "Component",
    "ComponentReference",
    "CrossSection",
    "Instance",
    "Path",
    "Pdk",
    "Port",
    "Section",
    "add_padding",
    "add_padding_container",
    "add_pins",
    "add_ports",
    "asserts",
    "boolean",
    "c",
    "call_if_func",
    "cell",
    "cell_without_validator",
    "components",
    "compose",
    "cross_section",
    "diff",
    "difftest",
    "get_active_pdk",
    "get_cell",
    "get_cells",
    "get_component",
    "get_constant",
    "get_cross_section",
    "get_cross_sections",
    "get_layer",
    "get_padding_points",
    "grid",
    "grid_with_text",
    "import_gds",
    "logger",
    "pack",
    "partial",
    "path",
    "port",
    "read",
    "show",
    "snap",
    "typings",
    "technology",
    "write_cells",
    "PATH",
    "kf",
    "kdb",
    "LayerEnum",
    "constant",
    "kcl",
)
