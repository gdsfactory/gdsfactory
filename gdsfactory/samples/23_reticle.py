"""Sample of a reticle top level Component."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    option = 3

    gc = gf.c.grating_coupler_elliptical_tm()
    c = gf.c.mzi_phase_shifter_top_heater_metal(delta_length=40)
    c = gf.routing.add_fiber_single(
        c, get_input_label_text_function=None, grating_coupler=gc
    )
    c = c.rotate(-90)

    if option == 1:
        c = gf.routing.add_electrical_pads_top(
            c, decorator=gf.add_labels.add_labels_to_ports
        )
    elif option == 2:
        c = gf.routing.add_electrical_pads_shortest(
            c, decorator=gf.add_labels.add_labels_to_ports
        )
    elif option == 3:
        c = gf.routing.add_electrical_pads_top_dc(
            c, decorator=gf.add_labels.add_labels_to_ports
        )
    c.show(show_ports=False)
