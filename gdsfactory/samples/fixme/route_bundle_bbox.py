import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    columns = 2
    ptop = c << gf.components.pad_array(columns=columns, port_orientation=270)
    pbot = c << gf.components.pad_array(port_orientation=90, columns=columns)
    ptop.movex(300)
    ptop.movey(300)

    obstacle = c << gf.c.rectangle(size=(300, 100), layer="M3")
    obstacle.ymin = pbot.ymax - 10 + 100
    obstacle.xmin = pbot.xmax - 10

    routes = gf.routing.route_bundle_electrical(
        c,
        pbot.ports,
        ptop.ports,
        start_straight_length=100,
        separation=20,
        bboxes=[
            obstacle.bbox(),
        ],
        sort_ports=True,
    )

    c.show()
