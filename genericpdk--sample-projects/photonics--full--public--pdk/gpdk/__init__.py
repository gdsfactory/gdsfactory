import typing
from functools import partial

import gdsfactory as gf
import kfactory as kf
from gdsfactory.config import PATH, __version__
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk
from gdsfactory.technology import LayerViews

from gpdk.cross_section import cross_sections
from gpdk.layer_map import LAYER
from gpdk.layer_stack import LAYER_STACK
from gpdk.models import get_models
from gpdk.routing_strategies import routing_strategies
from gpdk.simulation_settings import materials_index

if typing.TYPE_CHECKING:
    from gdsfactory.pdk import Pdk


PORT_MARKER_LAYER_TO_TYPE = {
    LAYER.PORT: "optical",
    LAYER.PORTE: "dc",
    LAYER.TE: "vertical_te",
    LAYER.TM: "vertical_tm",
}

PORT_LAYER_TO_TYPE = {
    LAYER.WG: "optical",
    LAYER.WGN: "optical",
    LAYER.SLAB150: "optical",
    LAYER.M1: "dc",
    LAYER.M2: "dc",
    LAYER.M3: "dc",
    LAYER.TE: "vertical_te",
    LAYER.TM: "vertical_tm",
}

PORT_TYPE_TO_MARKER_LAYER = {v: k for k, v in PORT_MARKER_LAYER_TO_TYPE.items()}


LAYER_CONNECTIVITY = [
    ("NPP", "VIAC", "M1"),
    ("PPP", "VIAC", "M1"),
    ("M1", "VIA1", "M2"),
    ("M2", "VIA2", "M3"),
]


LAYER_VIEWS = LayerViews(filepath=PATH.klayout_yaml)

_cells = get_cells([gf.components, gf.containers])

layer_transitions = {
    LAYER.WG: partial(gf.c.taper, cross_section="strip", length=10),
    (LAYER.WG, LAYER.WGN): "taper_sc_nc",
    (LAYER.WGN, LAYER.WG): "taper_nc_sc",
    LAYER.M3: "taper_electrical",
}
gf.kcl.layers = LAYER
gf.kcl.infos = kf.LayerInfos(
    **{v.name: kf.kdb.LayerInfo(v.layer, v.datatype) for v in LAYER},  # type: ignore[attr-defined]
)


class GenericConstants(gf.Constants):
    """Generic PDK constants."""

    fiber_input_to_output_spacing: float = 200.0
    metal_spacing: float = 10.0
    pad_pitch: float = 100.0
    pad_size: tuple[float, float] = (80.0, 80.0)
    wavelength: float = 1.55


PDK = Pdk(
    name="gpdk",
    version=__version__,
    cells=_cells,
    cross_sections=cross_sections,
    layers=LAYER,
    layer_stack=LAYER_STACK,
    layer_views=LAYER_VIEWS,
    layer_transitions=layer_transitions,  # type: ignore[arg-type]
    materials_index=materials_index,  # type: ignore[arg-type]
    constants=GenericConstants(),
    connectivity=LAYER_CONNECTIVITY,
    models=get_models(),
    routing_strategies=routing_strategies,
)

__all__ = ["LAYER", "LAYER_STACK"]
