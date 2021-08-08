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
from phidl import quickplot as plot
import phidl.geometry as pg
from phidl.device_layout import Group, Path, CrossSection

# NOTE: import order matters. Only change the order if you know what you are doing
from gdsfactory.component import Component, ComponentReference, copy
from gdsfactory.config import CONFIG, call_if_func
from gdsfactory.port import Port
from gdsfactory.port import port_array
from gdsfactory.cell import cell
from gdsfactory.cell import cell_without_validator
from gdsfactory.cell import clear_cache
from gdsfactory.tech import LAYER
from gdsfactory.cross_section import cross_section
from gdsfactory.show import show
from gdsfactory.write_doe import write_doe
from gdsfactory.import_gds import import_gds

import gdsfactory.asserts as asserts
import gdsfactory.components as components
import gdsfactory.routing as routing
import gdsfactory.bias as bias
import gdsfactory.klive as klive
import gdsfactory.port as port
import gdsfactory.types as types
import gdsfactory.path as path
import gdsfactory.snap as snap
import gdsfactory.tech as tech
import gdsfactory.containers as containers
import gdsfactory.components.extension as extend
import gdsfactory.component_from as component_from
import gdsfactory.read as read
import gdsfactory.remove as remove
import gdsfactory.layers as layers


from gdsfactory.tech import TECH
from gdsfactory.component_from_yaml import component_from_yaml
from gdsfactory.add_termination import add_termination
from gdsfactory.add_padding import (
    add_padding,
    get_padding_points,
    add_padding_container,
)
from gdsfactory.add_pins import add_pins, add_pins_to_references
from gdsfactory.fill import fill_rectangle
from gdsfactory.pack import pack
from gdsfactory.grid import grid
from gdsfactory.offset import offset
from gdsfactory.boolean import boolean
from gdsfactory.rotate import rotate


c = components
lys = layers.load_lyp_generic()

__all__ = [
    "CONFIG",
    "Component",
    "ComponentReference",
    "CrossSection",
    "Group",
    "LAYER",
    "Path",
    "Port",
    "TECH",
    "add_padding",
    "add_padding_container",
    "add_pins",
    "add_pins_to_references",
    "add_termination",
    "asserts",
    "bias",
    "boolean",
    "c",
    "call_if_func",
    "cell",
    "cell_without_validator",
    "clear_cache",
    "component_from",
    "component_from_yaml",
    "components",
    "containers",
    "copy",
    "cross_section",
    "extend",
    "fill_rectangle",
    "get_padding_points",
    "grid",
    "import_gds",
    "klive",
    "layers",
    "offset",
    "pack",
    "path",
    "partial",
    "pg",
    "plot",
    "port",
    "port_array",
    "read",
    "remove",
    "rotate",
    "routing",
    "show",
    "snap",
    "tech",
    "types",
    "write_doe",
]
__version__ = "3.0.0"


if __name__ == "__main__":
    print(__all__)
