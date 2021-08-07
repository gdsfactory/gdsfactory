import picwriter.components as pc
from picwriter.components.waveguide import WaveguideTemplate

from gdsfactory.config import TECH
from gdsfactory.types import Layer


def strip(
    wg_width: float = TECH.waveguide.strip.width,
    layer: Layer = TECH.waveguide.strip.layer,
    layer_cladding: Layer = (111, 0),
    radius: float = TECH.waveguide.strip.radius,
    cladding_offset: float = TECH.waveguide.strip.cladding_offset,
) -> WaveguideTemplate:

    return pc.WaveguideTemplate(
        bend_radius=radius,
        wg_width=wg_width,
        wg_layer=layer[0],
        wg_datatype=layer[1],
        clad_layer=layer_cladding[0],
        clad_datatype=layer_cladding[1],
        clad_width=cladding_offset,
        wg_type="strip",
    )


if __name__ == "__main__":
    c = strip()
