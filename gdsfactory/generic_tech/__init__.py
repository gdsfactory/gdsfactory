import pathlib
from functools import partial

from gdsfactory.generic_tech.layer_map import GenericLayerMap
from gdsfactory.generic_tech.layer_stack import get_layer_stack_generic
from gdsfactory.technology import LayerViews

module_path = pathlib.Path(__file__).parent.absolute()
layer_path = module_path / "klayout" / "tech" / "layers.lyp"

load_lyp_generic = partial(LayerViews.from_lyp, filepath=layer_path)

LAYER = GenericLayerMap()
LAYER_VIEWS = load_lyp_generic()
LAYER_STACK = get_layer_stack_generic()


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
