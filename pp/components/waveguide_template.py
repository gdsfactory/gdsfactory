import picwriter.components as pc
import pp


def wg_strip(
    wg_width=0.5,
    wg_layer=pp.LAYER.WG[0],
    wg_datatype=pp.LAYER.WG[1],
    clad_layer=pp.LAYER.WGCLAD[0],
    clad_datatype=pp.LAYER.WGCLAD[1],
    bend_radius=10,
):
    return pc.WaveguideTemplate(
        bend_radius=bend_radius,
        wg_width=wg_width,
        wg_layer=wg_layer,
        wg_datatype=wg_datatype,
        clad_layer=clad_layer,
        clad_datatype=clad_datatype,
    )
