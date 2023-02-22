from __future__ import annotations

import typing
from gdsfactory.generic_tech.layer_map import GenericLayerMap as LayerMap
from gdsfactory.generic_tech.layer_stack import LAYER_STACK
from gdsfactory.technology import LayerViews

if typing.TYPE_CHECKING:
    from gdsfactory.pdk import Pdk

LAYER = LayerMap()

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


def get_generic_pdk() -> Pdk:
    from gdsfactory.components import cells
    from gdsfactory.config import sparameters_path, PATH
    from gdsfactory.cross_section import cross_sections
    from gdsfactory.pdk import Pdk, constants

    LAYER_VIEWS = LayerViews(filepath=PATH.klayout_yaml)

    return Pdk(
        name="generic",
        cells=cells,
        cross_sections=cross_sections,
        layers=LAYER.dict(),
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        layer_transitions=LAYER_TRANSITIONS,
        sparameters_path=sparameters_path,
        constants=constants,
    )


def write_lyp() -> None:
    """Write from YAML to Klayout lyp file"""
    from gdsfactory.config import PATH

    LAYER_VIEWS = LayerViews(filepath=PATH.klayout_yaml)
    LAYER_VIEWS.to_lyp(PATH.klayout_lyp)


def write_yaml() -> None:
    """Write from YAML to Klayout lyp file"""
    from gdsfactory.config import PATH

    LAYER_VIEWS = LayerViews(filepath=PATH.klayout_lyp)
    LAYER_VIEWS.to_yaml(PATH.generic_tech / "layers2.yaml")


if __name__ == "__main__":
    # pdk = get_generic_pdk()
    # pdk.layer_views.to_yaml('layer_views2.yaml')
    # print(pdk.name)
    write_lyp()
    # write_yaml()
