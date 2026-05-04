"""PDK sample."""

from gdsfactory.config import CONF
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from mypdk import cells, config, tech
from mypdk.config import PATH
from mypdk.models import get_models
from mypdk.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

_models = get_models()
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)

CONF.pdk = "mypdk"
CONF.max_cellname_length = 64

__version__ = "0.0.0"


PDK = Pdk(
    name="mypdk",
    cells=_cells,
    cross_sections=_cross_sections,  # type: ignore
    layers=LAYER,
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
    models=_models,
    routing_strategies=routing_strategies,
)


__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "cells",
    "config",
    "tech",
]
