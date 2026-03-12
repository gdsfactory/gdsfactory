import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    p1 = c << gf.components.pad()
    p2 = c << gf.components.pad()
    p2.move((800, 2))
    gf.routing.route_bundle(
        c,
        [p1.ports["e3"]],
        [p2.ports["e1"]],
        cross_section="metal_routing",
        separation=10,
        sbend="bend_s",
        radius=10,
    )
    c.show()
