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

from phidl import quickplot as qp
import phidl.geometry as pg
import phidl.path as path
from phidl.device_layout import Group, Path, CrossSection

# NOTE: import order matters. Only change the order if you know what you are doing
from pp.assert_grating_coupler_properties import assert_grating_coupler_properties
from pp.config import CONFIG, call_if_func, conf
from pp.component import Component, ComponentReference
from pp.port import Port
from pp.port import port_array
from pp.cell import cell
from pp.cell import clear_cache
from pp.layers import LAYER
from pp.load_component import load_component

from pp.write_component import show
from pp.write_component import write_gds
from pp.write_component import write_component
from pp.write_doe import write_doe

import pp.components as c
import pp.routing as routing
import pp.bias as bias
import pp.klive as klive
import pp.sp as sp
import pp.port as port
import pp.types as types

from pp.component_from_yaml import component_from_yaml
from pp.recurse_references import recurse_references
from pp.types import get_name_to_function_dict

from pp.components.extension import extend_ports
from pp.add_termination import add_termination
from pp.add_padding import add_padding
from pp.add_pins import add_pins, add_pins_to_references
from pp.import_gds import import_gds
from pp.plotgds import plotgds
from pp.pack import pack
from pp.boolean import boolean
from pp.container import container


__all__ = [
    "assert_grating_coupler_properties",
    "container",
    "CONFIG",
    "LAYER",
    "Component",
    "ComponentReference",
    "CrossSection",
    "Group",
    "Path",
    "bias",
    "cell",
    "add_padding",
    "add_pins",
    "add_pins_to_references",
    "add_termination",
    "import_gds",
    "c",
    "clear_cache",
    "conf",
    "call_if_func",
    "extend_ports",
    "boolean",
    "get_name_to_function_dict",
    "klive",
    "load_component",
    "plotgds",
    "pack",
    "qp",
    "path",
    "pg",
    "port",
    "port_array",
    "routing",
    "recurse_references",
    "show",
    "sp",
    "types",
    "write_component",
    "write_doe",
    "write_gds",
    "Port",
    "component_from_yaml",
]
__version__ = "2.2.9"


if __name__ == "__main__":
    # print(__all__)
    print(get_name_to_function_dict(container, boolean))
