""" pp Photonics package provides some GDS useful

functions:

    - pp.show(): writes and shows the GDS in Klayout using klive
    - pp.plotgds(): plots GDS in matplotlib (good for notebooks)
    - pp.import_gds(): returns a component from a GDS

classes:

    - pp.Component
    - pp.Port
    - CONFIG

modules:

    - c: components
    - routing
    - Klive: send to klayout
    - layer: use layers
"""

# NOTE: import order matters. Only change the order if you know what you are doing
from pp.config import CONFIG
from pp.config import call_if_func
from pp.component import Component
from pp.component import ComponentReference
from pp.component import Port
from pp.name import autoname
from pp.name import autoname2
from pp.layers import LAYER
from pp.layers import layer
from pp.layers import preview_layerset
from pp.load_component import load_component
from pp.load_csv import load_csv

from pp.write_component import get_component_type
from pp.write_component import show
from pp.write_component import write_gds
from pp.write_component import write_component_type
from pp.write_component import write_component
from pp.write_doe import write_doe

import pp.components as c
import pp.routing as routing
import pp.bias as bias
import pp.klive as klive

from pp.components import component_factory
from pp.components.extension import extend_port
from pp.components.extension import extend_ports
from pp.add_padding import add_padding
from pp.import_gds import import_gds

from phidl import quickplot as plotgds


__all__ = [
    "CONFIG",
    "LAYER",
    "Component",
    "ComponentReference",
    "bias",
    "autoname",
    "autoname2",
    "add_padding",
    "import_gds",
    "c",
    "component_factory",
    "call_if_func",
    "extend_port",
    "extend_ports",
    "get_component_type",
    "klive",
    "layer",
    "load_component",
    "load_csv",
    "plotgds",
    "preview_layerset",
    "routing",
    "show",
    "write_component",
    "write_component_type",
    "write_doe",
    "write_gds",
    "Port",
]
__version__ = "1.1.1"


if __name__ == "__main__":
    print(__all__)
