"""Main module for gdsfactory."""

# NOTE: import order matters. Only change the order if you know what you are doing
# isort: skip_file

from __future__ import annotations
import sys
import warnings
from functools import partial
from toolz import compose
from aenum import constant

import kfactory as kf
from kfactory import LayerEnum, show, Instance
from kfactory.layout import kcl
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
    add_padding_to_size,
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


# Check Python version and issue a warning if using Python 3.10
if sys.version_info[:2] == (3, 10):
    warnings.warn(
        "Support for Python 3.10 has been dropped. Please upgrade to Python 3.11 or later "
        "to continue using the latest features and improvements. "
        "To get the latest gdsfactory, upgrading your Python version is required.",
        DeprecationWarning,
    )


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
    "add_padding_to_size",
    "add_pins",
    "add_ports",
    "boolean",
    "c",
    "cell",
    "clear_cache",
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
