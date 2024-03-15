from __future__ import annotations

import typing
from functools import cache

import kfactory as kf

from gdsfactory.config import PATH
from gdsfactory.generic_tech.layer_map import LAYER
from gdsfactory.generic_tech.layer_stack import LAYER_STACK
from gdsfactory.technology import LayerViews

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

LAYER_TRANSITIONS = {
    LAYER.WG: "taper",
    LAYER.M3: "taper",
    # (LAYER.)
}


@cache
def get_generic_pdk() -> Pdk:
    from gdsfactory.components import cells
    from gdsfactory.config import PATH
    from gdsfactory.cross_section import cross_sections
    from gdsfactory.generic_tech.containers import containers
    from gdsfactory.generic_tech.simulation_settings import materials_index
    from gdsfactory.pdk import Pdk, constants

    LAYER_VIEWS = LayerViews(filepath=PATH.klayout_yaml)

    cells = cells.copy()
    cells.update(containers)

    enclosure_rc = kf.LayerEnclosure(
        dsections=[(LAYER.SLAB90, 3.0)],
        main_layer=LAYER.SLAB90,
        name="enclosure_rc",
        kcl=kf.kcl,
    )
    kf.kcl.layer_enclosures = kf.kcell.LayerEnclosureModel(
        enclosure_map=dict(enclosure_rc=enclosure_rc)
    )

    kf.kcl.enclosure = kf.KCellEnclosure(
        enclosures=[enclosure_rc],
    )

    return Pdk(
        name="generic",
        cells=cells,
        cross_sections=cross_sections,
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        layer_transitions=LAYER_TRANSITIONS,
        materials_index=materials_index,
        constants=constants,
    )


if __name__ == "__main__":
    layer_views = LayerViews(filepath=PATH.klayout_yaml)
    layer_views.to_lyp(PATH.klayout_lyp)

    pdk = get_generic_pdk()
    # pdk.layer_views.to_yaml('layer_views2.yaml')
    # print(pdk.name)
