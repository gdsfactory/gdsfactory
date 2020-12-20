import picwriter.components as pc
from picwriter.components.waveguide import WaveguideTemplate

from pp.layers import LAYER


def wg_strip(
    wg_width: float = 0.5,
    wg_layer: int = LAYER.WG[0],
    wg_datatype: int = LAYER.WG[1],
    clad_layer: int = LAYER.WGCLAD[0],
    clad_datatype: int = LAYER.WGCLAD[1],
    bend_radius: int = 10,
    clad_offset: int = 3,
) -> WaveguideTemplate:
    return pc.WaveguideTemplate(
        bend_radius=bend_radius,
        wg_width=wg_width,
        wg_layer=wg_layer,
        wg_datatype=wg_datatype,
        clad_layer=clad_layer,
        clad_datatype=clad_datatype,
        clad_width=clad_offset,
        wg_type="strip",
    )


def wg_rib(
    wg_width: float = 0.5,
    wg_layer: int = LAYER.WG[0],
    wg_datatype: int = LAYER.WG[1],
    clad_layer: int = LAYER.SLAB90[0],
    clad_datatype: int = LAYER.SLAB90[1],
    bend_radius: int = 10,
    clad_offset: int = 3,
) -> WaveguideTemplate:
    return pc.WaveguideTemplate(
        bend_radius=bend_radius,
        wg_width=wg_width,
        wg_layer=wg_layer,
        wg_datatype=wg_datatype,
        clad_layer=clad_layer,
        clad_datatype=clad_datatype,
        clad_width=clad_offset,
        wg_type="strip",
    )


if __name__ == "__main__":
    c = wg_strip()
