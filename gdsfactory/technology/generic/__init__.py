import pathlib
from functools import partial

from gdsfactory.technology.generic.layer_map import LAYER
from gdsfactory.technology.generic.layer_stack import LAYER_STACK
from gdsfactory.technology.layer_views import LayerViews

module_path = pathlib.Path(__file__).parent.absolute()
layer_path = module_path / "klayout" / "tech" / "layers.lyp"

load_lyp_generic = partial(LayerViews.from_lyp, filepath=layer_path)

try:
    LAYER_VIEWS = load_lyp_generic()
except Exception:
    print(f"Error loading generic layermap in {layer_path!r}")
    LAYER_VIEWS = LayerViews()


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


__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PORT_MARKER_LAYER_TO_TYPE",
    "PORT_LAYER_TO_TYPE",
    "PORT_TYPE_TO_MARKER_LAYER",
]
