import picwriter.components as pc
import pp
from picwriter.components.waveguide import WaveguideTemplate


def wg_strip(
    wg_width: float = 0.5,
    wg_layer: int = pp.LAYER.WG[0],
    wg_datatype: int = pp.LAYER.WG[1],
    clad_layer: int = pp.LAYER.WGCLAD[0],
    clad_datatype: int = pp.LAYER.WGCLAD[1],
    bend_radius: int = 10,
    cladding_offset: int = 3,
) -> WaveguideTemplate:
    return pc.WaveguideTemplate(
        bend_radius=bend_radius,
        wg_width=wg_width,
        wg_layer=wg_layer,
        wg_datatype=wg_datatype,
        clad_layer=clad_layer,
        clad_datatype=clad_datatype,
        clad_width=cladding_offset,
    )
