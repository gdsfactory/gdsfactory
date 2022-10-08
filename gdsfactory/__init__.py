"""You can import gdsfactory.as gf.

functions:
    - import_gds(): returns a Component from a GDS

classes:

    - Component
    - Port
    - CONFIG
    - TECH

modules:

    - c: components
    - routing

isort:skip_file
"""
from functools import partial
from toolz import compose
from gdsfactory.component_layout import Group
from gdsfactory.path import Path


# NOTE: import order matters. Only change the order if you know what you are doing
from gdsfactory.component import Component, ComponentReference
from gdsfactory.config import CONFIG, CONF, call_if_func
from gdsfactory.port import Port
from gdsfactory.cell import cell
from gdsfactory.cell import cell_without_validator
from gdsfactory.cell import clear_cache
from gdsfactory.tech import LAYER
from gdsfactory.show import show
from gdsfactory.read.import_gds import import_gds
from gdsfactory.cross_section import CrossSection, Section
from gdsfactory.types import Label

from gdsfactory import decorators
from gdsfactory import cross_section
from gdsfactory import labels
from gdsfactory import asserts
from gdsfactory import components
from gdsfactory import routing
from gdsfactory import types
from gdsfactory import path
from gdsfactory import snap
from gdsfactory import tech
from gdsfactory import read
from gdsfactory import layers
from gdsfactory import add_termination
from gdsfactory import functions
from gdsfactory import export
from gdsfactory import geometry
from gdsfactory import add_ports
from gdsfactory import write_cells
from gdsfactory import add_pins

from gdsfactory.tech import TECH
from gdsfactory.add_tapers import add_tapers
from gdsfactory.add_padding import (
    add_padding,
    add_padding_container,
    get_padding_points,
)
from gdsfactory.fill import fill_rectangle
from gdsfactory.pack import pack
from gdsfactory.grid import grid, grid_with_text
from gdsfactory.pdk import (
    Pdk,
    get_component,
    get_cross_section,
    get_layer,
    get_active_pdk,
    get_cell,
)
from gdsfactory.get_factories import get_cells
from gdsfactory.cross_section import get_cross_section_factories


c = components

__all__ = (
    "CONF",
    "CONFIG",
    "Component",
    "ComponentReference",
    "CrossSection",
    "Group",
    "LAYER",
    "Label",
    "Path",
    "Pdk",
    "Port",
    "Section",
    "TECH",
    "add_padding",
    "add_padding_container",
    "add_pins",
    "add_ports",
    "add_tapers",
    "add_termination",
    "asserts",
    "c",
    "call_if_func",
    "cell",
    "cell_without_validator",
    "clear_cache",
    "components",
    "compose",
    "cross_section",
    "decorators",
    "export",
    "fill_rectangle",
    "functions",
    "geometry",
    "get_active_pdk",
    "get_cell",
    "get_cells",
    "get_component",
    "get_cross_section",
    "get_cross_section_factories",
    "get_layer",
    "get_padding_points",
    "grid",
    "grid_with_text",
    "import_gds",
    "labels",
    "layers",
    "pack",
    "partial",
    "path",
    "read",
    "routing",
    "show",
    "snap",
    "tech",
    "types",
    "write_cells",
)
__version__ = "5.39.0"
