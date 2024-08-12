"""Main module for gdsfactory."""

# NOTE: import order matters. Only change the order if you know what you are doing
# isort: skip_file

from __future__ import annotations
from functools import partial
from toolz import compose
from aenum import constant  # type: ignore[import-untyped]

import kfactory as kf
from kfactory.kcell import LayerEnum, kcl, show, vcell
from kfactory import logger
import klayout.db as kdb

from gdsfactory.cell import cell
from gdsfactory.path import Path
from gdsfactory.component import (
    Component,
    ComponentBase,
    ComponentAllAngle,
    ComponentReference,
    Instance,
    container,
    component_with_function,
)
from gdsfactory.config import CONF, PATH
from gdsfactory.port import Port
from gdsfactory.read.import_gds import import_gds
from gdsfactory.cross_section import CrossSection, Section, xsection
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
    get_layer_name,
    get_active_pdk,
    get_cell,
    get_constant,
)
from gdsfactory.get_factories import get_cells
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.grid import grid, grid_with_text

c = components


def clear_cache(kcl: kf.KCLayout = kf.kcl) -> None:
    """Clears the whole layout object cache for the default layout."""
    kcl.clear_kcells()


__all__ = (
    "CONF",
    "Component",
    "ComponentAllAngle",
    "ComponentBase",
    "ComponentReference",
    "CrossSection",
    "Instance",
    "LayerEnum",
    "PATH",
    "Path",
    "Pdk",
    "Port",
    "Section",
    "add_padding",
    "add_padding_container",
    "add_pins",
    "add_ports",
    "boolean",
    "c",
    "clear_cache",
    "cell",
    "components",
    "compose",
    "constant",
    "container",
    "containers",
    "component_with_function",
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
    "get_padding_points",
    "grid",
    "grid_with_text",
    "import_gds",
    "kcl",
    "kdb",
    "kf",
    "logger",
    "labels",
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
