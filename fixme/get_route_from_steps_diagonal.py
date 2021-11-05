"""
FIXME, shall we allow diagonal route for electrical connections?

"""

if __name__ == "__main__":
    import gdsfactory as gf

    c = gf.Component("pads_route_from_steps")
    pt = c << gf.c.pad_array(orientation=270, columns=3)
    pb = c << gf.c.pad_array(orientation=90, columns=3)
    pt.move((100, 200))
    route = gf.routing.get_route_from_steps(
        pt.ports["e11"],
        pb.ports["e11"],
        steps=[
            {"y": 100},
        ],
        cross_section=gf.cross_section.metal3,
        bend=gf.components.wire_corner,
    )
    c.add(route.references)
    c.show()
