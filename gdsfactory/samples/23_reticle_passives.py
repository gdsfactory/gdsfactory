"""
Sample of a reticle
"""

import gdsfactory as gf


def mzi_te(**kwargs):
    gc = gf.c.grating_coupler_elliptical_tm()
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(
        c, get_input_label_text_function=None, grating_coupler=gc
    )
    gf.dft.add_label_yaml(c)
    c = c.rotate(-90)
    return c


if __name__ == "__main__":
    c = gf.grid(
        [
            mzi_te(),
            mzi_te(),
            gf.functions.rotate(mzi_te),
            gf.dft.add_label_yaml(
                gf.functions.mirror(gf.routing.add_fiber_single(gf.components.mmi1x2))
            ),
        ]
    )
    gdspath = c.write_gds("mask.gds")
    csvpath = gf.mask.write_labels_gdspy(gdspath, prefix="component_name")
    c.show()
