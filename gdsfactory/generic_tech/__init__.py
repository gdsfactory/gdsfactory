from gdsfactory.config import layer_path
from gdsfactory.generic_tech.layer_map import GenericLayerMap as LayerMap
from gdsfactory.generic_tech.layer_stack import LAYER_STACK
from gdsfactory.technology import LayerViews

LAYER = LayerMap()
LAYER_VIEWS = LayerViews(filepath=layer_path)

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


def get_generic_pdk():
    from gdsfactory.components import cells
    from gdsfactory.config import sparameters_path
    from gdsfactory.cross_section import cross_sections
    from gdsfactory.pdk import Pdk, constants

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


if __name__ == "__main__":
    pdk = get_generic_pdk()
    print(pdk.name)
