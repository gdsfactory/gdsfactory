import gdsfactory as gf

straight = gf.c.straight()
bend = gf.c.bend_euler()


component_factories = dict(straight=straight, bend=bend)
