""" You can import gdsfactory.as gf

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
from phidl.device_layout import Group
from gdsfactory.quickplotter import quickplot, quickplot2, set_quickplot_options
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

from gdsfactory import cross_section
from gdsfactory import asserts
from gdsfactory import components
from gdsfactory import dft
from gdsfactory import routing
from gdsfactory import klive
from gdsfactory import port
from gdsfactory import types
from gdsfactory import path
from gdsfactory import snap
from gdsfactory import tech
from gdsfactory import read
from gdsfactory import layers
from gdsfactory import add_termination
from gdsfactory import add_grating_couplers
from gdsfactory import functions
from gdsfactory import export
from gdsfactory import geometry
from gdsfactory import mask
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
plot = quickplot

__all__ = [
    "CONFIG",
    "CONF",
    "Component",
    "ComponentReference",
    "CrossSection",
    "Section",
    "compose",
    "Group",
    "LAYER",
    "Path",
    "Port",
    "TECH",
    "add_grating_couplers",
    "add_padding",
    "add_padding_container",
    "add_pins",
    "add_ports",
    "add_tapers",
    "add_termination",
    "asserts",
    "geometry",
    "c",
    "call_if_func",
    "cell",
    "cell_without_validator",
    "clear_cache",
    "components",
    "cross_section",
    "dft",
    "export",
    "fill_rectangle",
    "functions",
    "get_padding_points",
    "grid",
    "grid_with_text",
    "import_gds",
    "klive",
    "layers",
    "mask",
    "pack",
    "path",
    "partial",
    "plot",
    "port",
    "read",
    "routing",
    "show",
    "snap",
    "tech",
    "types",
    "write_cells",
    "Label",
    "Pdk",
    "get_active_pdk",
    "get_component",
    "get_cross_section",
    "get_cell",
    "get_cells",
    "get_layer",
    "get_cross_section_factories",
    "quickplot",
    "quickplot2",
    "set_quickplot_options",
]
__version__ = "5.9.0"
