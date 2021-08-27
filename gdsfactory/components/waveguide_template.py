import picwriter.components as pc
from picwriter.components.waveguide import WaveguideTemplate

from gdsfactory.types import Layer


def strip(
    wg_width: float = 0.5,
    layer: Layer = (1, 0),
    layer_cladding: Layer = (111, 0),
    radius: float = 10.0,
    cladding_offset: float = 3.0,
    euler_bend: bool = True,
    wg_type: str = "strip",
) -> WaveguideTemplate:
    """
    wg_type: strip, slot, and swg (subwavelength)
    resist: Specifies the type of photoresist used (+ or -)
    """

    return pc.WaveguideTemplate(
        bend_radius=radius,
        wg_width=wg_width,
        wg_layer=layer[0],
        wg_datatype=layer[1],
        clad_layer=layer_cladding[0],
        clad_datatype=layer_cladding[1],
        clad_width=cladding_offset,
        wg_type=wg_type,
        euler_bend=euler_bend,
    )


if __name__ == "__main__":
    c = strip()
