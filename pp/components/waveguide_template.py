import picwriter.components as pc
from picwriter.components.waveguide import WaveguideTemplate

from pp.config import conf
from pp.layers import LAYER
from pp.types import Layer


def wg_strip(
    wg_width: float = 0.5,
    layer: Layer = LAYER.WG,
    layer_cladding: Layer = LAYER.WGCLAD,
    bend_radius: float = 10.0,
    cladding_offset: float = conf.tech.cladding_offset,
) -> WaveguideTemplate:

    return pc.WaveguideTemplate(
        bend_radius=bend_radius,
        wg_width=wg_width,
        wg_layer=layer[0],
        wg_datatype=layer[1],
        clad_layer=layer_cladding[0],
        clad_datatype=layer_cladding[1],
        clad_width=cladding_offset,
        wg_type="strip",
    )


if __name__ == "__main__":
    c = wg_strip()
