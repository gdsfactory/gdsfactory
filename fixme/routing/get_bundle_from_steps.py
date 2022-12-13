"""FIXME, only first route looks good."""


if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("pads_bundle_from_steps")
    pt = c << gf.components.pad_array(orientation=270, columns=3)
    pb = c << gf.components.pad_array(orientation=90, columns=3)
    pt.move((100, 200))

    routes = gf.routing.get_bundle_from_steps(
        pt.ports,
        pb.ports,
        cross_section="metal_routing",
        bend=gf.components.wire_corner,
        steps=[{"y": 100}, {"x": 100}],
    )

    for route in routes:
        c.add(route.references)
    c.show(show_ports=True)
