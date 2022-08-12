"""FIXME: electrical connections should ignore port orientation."""

import gdsfactory as gf


if __name__ == "__main__":

    c = gf.Component("mzi_with_pads")
    c1 = c << gf.components.mzi_phase_shifter_top_heater_metal(length_x=70)
    c2 = c << gf.components.pad_array90(columns=2)

    c2.ymin = c1.ymax + 20
    c2.x = 0
    c1.x = 0

    ports1 = c1.get_ports_list(width=11)
    ports2 = c2.get_ports_list()

    routes = gf.routing.get_bundle(
        ports1=ports1,
        ports2=ports2,
        cross_section=gf.cross_section.metal1,
        width=5,
        bend=gf.components.wire_corner,
    )
    for route in routes:
        c.add(route.references)

    c.show(show_ports=True)
