""" pp Photonics package

functions:

    - pp.show(): writes and shows the GDS in Klayout using klive
    - pp.import_gds(): returns a Component from a GDS

classes:

    - pp.Component
    - pp.Port
    - CONFIG

modules:

    - c: components
    - routing
    - layer: GDS layers

isort:skip_file
"""

from phidl import quickplot as plot
import phidl.geometry as pg
from phidl.device_layout import Group, Path, CrossSection

# NOTE: import order matters. Only change the order if you know what you are doing
from pp.assert_grating_coupler_properties import assert_grating_coupler_properties
from pp.config import CONFIG, call_if_func
from pp.component import Component, ComponentReference, copy
from pp.port import Port
from pp.port import port_array
from pp.cell import cell
from pp.cell import cell_with_validator
from pp.cell import clear_cache
from pp.layers import LAYER
from pp.load_component import load_component
from pp.cross_section import cross_section

from pp.show import show
from pp.write_doe import write_doe

import pp.assert_version as assert_version
import pp.components as components
import pp.routing as routing
import pp.bias as bias
import pp.klive as klive
import pp.port as port
import pp.types as types
import pp.path as path
from pp.tech import TECH

from pp.component_from_yaml import component_from_yaml

from pp.components.extension import extend_ports
from pp.add_termination import add_termination
from pp.add_padding import add_padding, get_padding_points, add_padding_container
from pp.add_pins import add_pins, add_pins_to_references
from pp.import_gds import import_gds
from pp.fill import fill_rectangle
from pp.pack import pack
from pp.grid import grid
from pp.offset import offset
from pp.boolean import boolean
from pp.rotate import rotate
from pp.snap import snap_to_grid
from pp.set_plot_options import set_plot_options


c = components

__all__ = [
    "assert_grating_coupler_properties",
    "assert_version",
    "CONFIG",
    "LAYER",
    "Component",
    "ComponentReference",
    "CrossSection",
    "cross_section",
    "Group",
    "Path",
    "bias",
    "cell",
    "cell_with_validator",
    "copy",
    "add_padding",
    "add_padding_container",
    "add_pins",
    "add_pins_to_references",
    "add_termination",
    "fill_rectangle",
    "import_gds",
    "c",
    "components",
    "clear_cache",
    "call_if_func",
    "extend_ports",
    "boolean",
    "types",
    "get_padding_points",
    "grid",
    "klive",
    "load_component",
    "offset",
    "pack",
    "plot",
    "path",
    "pg",
    "port",
    "port_array",
    "rotate",
    "routing",
    "show",
    "snap_to_grid",
    "set_plot_options",
    "write_doe",
    "Port",
    "component_from_yaml",
    "TECH",
]
__version__ = "2.5.0"


if __name__ == "__main__":
    print(__all__)
