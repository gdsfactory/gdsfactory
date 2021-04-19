""" pp Photonics package

functions:

    - pp.show(): writes and shows the GDS in Klayout using klive
    - pp.plotgds(): plots GDS in matplotlib
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
from pp.config import CONFIG, call_if_func, conf
from pp.component import Component, ComponentReference, copy
from pp.port import Port
from pp.port import port_array
from pp.cell import cell
from pp.cell import clear_cache
from pp.layers import LAYER
from pp.load_component import load_component

from pp.show import show
from pp.write_doe import write_doe

import pp.components as components
import pp.routing as routing
import pp.bias as bias
import pp.klive as klive
import pp.port as port
import pp.types as types
import pp.path as path
import pp.cross_section as cross_section

from pp.component_from_yaml import component_from_yaml
from pp.types import get_name_to_function_dict

from pp.components.extension import extend_ports
from pp.add_termination import add_termination
from pp.add_padding import add_padding, get_padding_points, add_padding_container
from pp.add_pins import add_pins, add_pins_to_references
from pp.import_gds import import_gds
from pp.plotgds import plotgds
from pp.pack import pack
from pp.grid import grid
from pp.offset import offset
from pp.boolean import boolean
from pp.rotate import rotate
from pp.snap import snap_to_grid


c = components

__all__ = [
    "assert_grating_coupler_properties",
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
    "copy",
    "add_padding",
    "add_padding_container",
    "add_pins",
    "add_pins_to_references",
    "add_termination",
    "import_gds",
    "c",
    "components",
    "clear_cache",
    "conf",
    "call_if_func",
    "extend_ports",
    "boolean",
    "get_name_to_function_dict",
    "get_padding_points",
    "grid",
    "klive",
    "load_component",
    "offset",
    "plotgds",
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
    "types",
    "write_doe",
    "Port",
    "component_from_yaml",
]
__version__ = "2.4.7"


if __name__ == "__main__":
    print(__all__)
