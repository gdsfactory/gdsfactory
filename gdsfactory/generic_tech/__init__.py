from gdsfactory.config import layer_path
from gdsfactory.generic_tech.layer_map import GenericLayerMap
from gdsfactory.generic_tech.layer_stack import get_layer_stack_generic
from gdsfactory.technology import LayerViews

LAYER = GenericLayerMap()
LAYER_VIEWS = LayerViews.from_lyp(filepath=layer_path)
LAYER_STACK = get_layer_stack_generic()

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


def get_generic_pdk():
    from gdsfactory.components import cells
    from gdsfactory.config import sparameters_path
    from gdsfactory.cross_section import cross_sections
    from gdsfactory.pdk import Pdk

    return Pdk(
        name="generic",
        cells=cells,
        cross_sections=cross_sections,
        layers=LAYER.dict(),
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        sparameters_path=sparameters_path,
    )


if __name__ == "__main__":
    pdk = get_generic_pdk()
    print(pdk.name)
