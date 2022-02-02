import gdsfactory as gf

straight = gf.components.straight()
bend = gf.components.bend_euler()


component_factories = dict(straight=straight, bend=bend)
