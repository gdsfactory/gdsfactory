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
from phidl.device_layout import Group, Path

# NOTE: import order matters. Only change the order if you know what you are doing
from gdsfactory.filecache import filecache
from gdsfactory.component import Component, ComponentReference, copy
from gdsfactory.config import CONFIG, call_if_func
from gdsfactory.port import Port
from gdsfactory.port import port_array
from gdsfactory.cell import cell
from gdsfactory.cell import cell_without_validator
from gdsfactory.cell import clear_cache
from gdsfactory.tech import LAYER
from gdsfactory.show import show
from gdsfactory.write_doe import write_doe
from gdsfactory.import_gds import import_gds
from gdsfactory.cross_section import CrossSection

from gdsfactory import cross_section
from gdsfactory import asserts
from gdsfactory import components
from gdsfactory import routing
from gdsfactory import bias
from gdsfactory import klive
from gdsfactory import port
from gdsfactory import types
from gdsfactory import path
from gdsfactory import snap
from gdsfactory import tech
from gdsfactory import containers
from gdsfactory import component_from
from gdsfactory import read
from gdsfactory import remove
from gdsfactory import layers
from gdsfactory import add_termination


from gdsfactory.tech import TECH
from gdsfactory.component_from_yaml import component_from_yaml
from gdsfactory.add_tapers import add_tapers
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
    "add_tapers",
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
    "fill_rectangle",
    "filecache",
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
__version__ = "3.1.8"


if __name__ == "__main__":
    print(__all__)
