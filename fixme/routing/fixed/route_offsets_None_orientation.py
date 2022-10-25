"""get_bundle_from_steps does not work ports with None orientation. FIXME."""


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("pads_bundle_steps")
    pt = c << gf.components.pad_array(
        gf.partial(gf.components.pad, size=(30, 30)),
        orientation=None,
        columns=3,
        spacing=(50, 0),
    )
    pb = c << gf.components.pad_array(orientation=None, columns=1, rows=3)
    pt.move((500, 500))

    routes = gf.routing.get_bundle_from_steps_electrical(
        pt.ports,
        pb.ports,
        end_straight_length=60,
        separation=30,
        steps=[{"dy": -50}, {"dx": -100}],
    )

    for route in routes:
        c.add(route.references)

    c.show(show_ports=True)
