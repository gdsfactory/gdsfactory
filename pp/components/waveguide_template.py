import picwriter.components as pc


def wg_strip(
    wg_width=0.5,
    wg_layer=1,
    wg_datatype=0,
    clad_layer=2,
    clad_datatype=0,
    bend_radius=50,
):
    return pc.WaveguideTemplate(
        bend_radius=50.0,
        wg_width=wg_width,
        wg_layer=wg_layer,
        wg_datatype=wg_datatype,
        clad_layer=clad_layer,
        clad_datatype=clad_datatype,
    )
