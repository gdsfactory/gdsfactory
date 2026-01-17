from __future__ import annotations

import typing
from functools import cache, partial

import kfactory as kf

from gdsfactory.config import PATH
from gdsfactory.gpdk.layer_map import LAYER
from gdsfactory.gpdk.layer_stack import LAYER_STACK
from gdsfactory.technology import LayerViews
from gdsfactory.typings import RoutingStrategy

if typing.TYPE_CHECKING:
    from gdsfactory.pdk import Pdk

    PDK: Pdk

__all__ = ["LAYER", "LAYER_STACK", "get_generic_pdk"]


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


@cache
def get_generic_pdk() -> Pdk:
    import gdsfactory as gf
    from gdsfactory.config import __version__
    from gdsfactory.cross_section import cross_sections
    from gdsfactory.get_factories import get_cells
    from gdsfactory.gpdk.simulation_settings import materials_index
    from gdsfactory.pdk import GenericConstants, Pdk

    LAYER_VIEWS = LayerViews(filepath=PATH.klayout_yaml)

    cells = get_cells([gf.components])
    containers_dict = get_cells([gf.containers])

    layer_transitions = {
        LAYER.WG: "taper",
        (LAYER.WG, LAYER.WGN): "taper_sc_nc",
        (LAYER.WGN, LAYER.WG): "taper_nc_sc",
        LAYER.M3: "taper_electrical",
    }
    gf.kcl.layers = LAYER
    gf.kcl.infos = kf.LayerInfos(
        **{v.name: kf.kdb.LayerInfo(v.layer, v.datatype) for v in LAYER},  # type: ignore[attr-defined]
    )
    routing_strategies: dict[str, RoutingStrategy] = dict(
        route_bundle=partial(gf.routing.route_bundle, cross_section="strip"),
        route_bundle_all_angle=partial(
            gf.routing.route_bundle_all_angle, cross_section="strip"
        ),
        route_bundle_electrical=partial(
            gf.routing.route_bundle_electrical, cross_section="metal_routing"
        ),
    )

    return Pdk(
        name="generic",
        version=__version__,
        cells={c: cells[c] for c in cells if c not in containers_dict},  # type: ignore
        containers=containers_dict,  # type: ignore[arg-type]
        cross_sections=cross_sections,
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        layer_transitions=layer_transitions,  # type: ignore[arg-type]
        materials_index=materials_index,  # type: ignore[arg-type]
        constants=GenericConstants(),
        connectivity=LAYER_CONNECTIVITY,
        routing_strategies=routing_strategies,
    )


# Lazy PDK instantiation to avoid circular imports
_PDK = None


def _get_pdk() -> Pdk:
    global _PDK
    if _PDK is None:
        _PDK = get_generic_pdk()
    return _PDK


def __getattr__(name: str) -> Pdk:
    if name == "PDK":
        return _get_pdk()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
