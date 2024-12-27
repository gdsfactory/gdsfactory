"""Main module for gdsfactory."""

# NOTE: import order matters. Only change the order if you know what you are doing
# isort: skip_file

from __future__ import annotations
from functools import partial
from toolz import compose  # type: ignore
from aenum import constant  # type: ignore[import-untyped]

import kfactory as kf
from kfactory.kcell import LayerEnum, kcl, show, Instance
from kfactory import logger
import klayout.db as kdb

from gdsfactory._cell import cell, vcell
from gdsfactory.path import Path
from gdsfactory.component import (
    Component,
    ComponentBase,
    ComponentAllAngle,
    ComponentReference,
    container,
    component_with_function,
)
from gdsfactory.config import CONF, PATH, __version__
from gdsfactory.port import Port
from gdsfactory.read.import_gds import import_gds
from gdsfactory.cross_section import (
    ComponentAlongPath,
    CrossSection,
    Section,
    xsection,
    get_cross_sections,
)
from gdsfactory.difftest import difftest, diff
from gdsfactory.boolean import boolean

from gdsfactory import cross_section
from gdsfactory import port
from gdsfactory import components
from gdsfactory import containers
from gdsfactory import labels
from gdsfactory import typings
from gdsfactory import path
from gdsfactory import snap
from gdsfactory import read
from gdsfactory import add_ports
from gdsfactory import write_cells
from gdsfactory import add_pins
from gdsfactory import technology
from gdsfactory import routing
from gdsfactory import export
from gdsfactory import functions

from gdsfactory.add_padding import (
    add_padding,
    add_padding_container,
    get_padding_points,
)
from gdsfactory.pack import pack
from gdsfactory.pdk import (
    Pdk,
    get_component,
    get_cross_section,
    get_layer,
    get_layer_tuple,
    get_layer_name,
    get_active_pdk,
    get_cell,
    get_constant,
)
from gdsfactory.get_factories import get_cells
from gdsfactory.grid import grid, grid_with_text

c = components


def clear_cache(kcl: kf.KCLayout = kf.kcl) -> None:
    """Clears the whole layout object cache for the default layout."""
    kcl.clear_kcells()


__all__ = (
    "CONF",
    "PATH",
    "Component",
    "ComponentAllAngle",
    "ComponentAlongPath",
    "ComponentBase",
    "ComponentReference",
    "CrossSection",
    "Instance",
    "LayerEnum",
    "Path",
    "Pdk",
    "Port",
    "Section",
    "__version__",
    "add_padding",
    "add_padding_container",
    "add_pins",
    "add_ports",
    "boolean",
    "c",
    "cell",
    "clear_cache",
    "component_with_function",
    "components",
    "compose",
    "constant",
    "container",
    "containers",
    "cross_section",
    "diff",
    "difftest",
    "export",
    "functions",
    "get_active_pdk",
    "get_cell",
    "get_cells",
    "get_component",
    "get_constant",
    "get_cross_section",
    "get_cross_sections",
    "get_layer",
    "get_layer_name",
    "get_layer_tuple",
    "get_padding_points",
    "grid",
    "grid_with_text",
    "import_gds",
    "kcl",
    "kdb",
    "kf",
    "labels",
    "logger",
    "pack",
    "partial",
    "path",
    "port",
    "read",
    "routing",
    "show",
    "snap",
    "technology",
    "typings",
    "vcell",
    "write_cells",
    "xsection",
)
