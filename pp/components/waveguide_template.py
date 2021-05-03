import picwriter.components as pc
from picwriter.components.waveguide import WaveguideTemplate

from pp.config import TECH
from pp.types import Layer


def wg_strip(
    wg_width: float = TECH.waveguide.strip.width,
    layer: Layer = TECH.waveguide.strip.layer,
    layer_cladding: Layer = TECH.waveguide.strip.layers_cladding[0],
    bend_radius: float = TECH.waveguide.strip.bend_radius,
    cladding_offset: float = TECH.waveguide.strip.cladding_offset,
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
