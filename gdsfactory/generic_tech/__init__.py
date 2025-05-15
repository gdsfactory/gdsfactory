from __future__ import annotations

import typing
from functools import cache, partial

from gdsfactory.config import PATH
from gdsfactory.generic_tech.layer_map import LAYER
from gdsfactory.generic_tech.layer_stack import LAYER_STACK
from gdsfactory.technology import LayerViews

if typing.TYPE_CHECKING:
    from gdsfactory.pdk import Pdk

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
    from gdsfactory.config import PATH, __version__
    from gdsfactory.cross_section import cross_sections
    from gdsfactory.generic_tech.simulation_settings import materials_index
    from gdsfactory.get_factories import get_cells
    from gdsfactory.pdk import Pdk, constants

    LAYER_VIEWS = LayerViews(filepath=PATH.klayout_yaml)

    cells = get_cells([gf.components])
    containers_dict = get_cells([gf.containers])

    layer_transitions = {
        LAYER.WG: partial(gf.c.taper, cross_section="strip", length=10),
        (LAYER.WG, LAYER.WGN): "taper_sc_nc",
        (LAYER.WGN, LAYER.WG): "taper_nc_sc",
        LAYER.M3: "taper_electrical",
    }

    return Pdk(
        name="generic",
        version=__version__,
        cells={c: cells[c] for c in cells if c not in containers_dict},
        containers=containers_dict,
        cross_sections=cross_sections,
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        layer_transitions=layer_transitions,  # type: ignore[arg-type]
        materials_index=materials_index,  # type: ignore[arg-type]
        constants=constants,
        connectivity=LAYER_CONNECTIVITY,
    )


if __name__ == "__main__":
    from gdsfactory.technology.klayout_tech import KLayoutTechnology

    LAYER_VIEWS = LayerViews(filepath=PATH.klayout_yaml)
    connectivity = [
        ("HEATER", "VIA1", "M2"),
        ("M1", "VIA1", "M2"),
        ("M2", "VIA2", "M3"),
    ]

    t = KLayoutTechnology(
        name="generic_tech",
        layer_map=LAYER,  # type: ignore[arg-type]
        layer_views=LAYER_VIEWS,
        layer_stack=LAYER_STACK,
        connectivity=connectivity,
    )
    t.write_tech(tech_dir=PATH.klayout)

    layer_views = LayerViews(filepath=PATH.klayout_yaml)
    layer_views.to_lyp(PATH.klayout_lyp)

    pdk = get_generic_pdk()
    pdk.activate()
