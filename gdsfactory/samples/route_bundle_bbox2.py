import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    columns = 2
    ptop = c << gf.components.pad_array(columns=columns, port_orientation=270)
    pbot = c << gf.components.pad_array(port_orientation=270, columns=columns)
    ptop.dmovex(300)
    ptop.dmovey(300)

    obstacle = c << gf.c.rectangle(size=(300, 100), layer="M3")
    obstacle.dymin = pbot.dymax - 10
    obstacle.dxmin = pbot.dxmax - 10

    routes = gf.routing.route_bundle_electrical(
        c,
        pbot.ports,
        ptop.ports,
        start_straight_length=100,
        separation=20,
        bboxes=[
            obstacle.bbox(),
            pbot.bbox(),
            ptop.bbox(),
        ],  # obstacles to avoid
        sort_ports=True,
        cross_section=gf.cross_section.metal_routing,
    )

    c.show()
