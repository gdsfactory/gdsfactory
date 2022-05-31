import gdsfactory as gf

if __name__ == "__main__":

    c = gf.Component("mzi_with_pads")
    c1 = c << gf.components.mzi_phase_shifter(
        straight_x_top=gf.components.straight_heater_metal_90_90, length_x=70  # 150
    )
    c2 = c << gf.components.pad_array(columns=2, orientation=270)

    c2.ymin = c1.ymax + 30
    c2.x = 0
    c1.x = 0

    ports1 = c1.get_ports_list(port_type="electrical")
    ports2 = c2.get_ports_list()

    routes = gf.routing.get_bundle(
        ports1=ports1,
        ports2=ports2,
        cross_section=gf.cross_section.metal1,
        width=10,
        bend=gf.components.wire_corner,
    )
    for route in routes:
        c.add(route.references)

    c.show()
