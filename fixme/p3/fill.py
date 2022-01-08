"""phidl Dummy fill does not look as good as the one in klayout

Maybe: create klayout dummy fill example

"""

import toolz
import gdsfactory as gf

ring_te = toolz.compose(gf.routing.add_fiber_array, gf.c.ring_single)
rings = gf.grid([ring_te(radius=r) for r in [10, 20, 50]])


@gf.cell
def mask(size=(1000, 1000)):
    c = gf.Component()
    c << gf.components.die(size=size)
    c << rings
    c << gf.fill_rectangle(
        c, fill_layers=[(2, 0)], fill_densities=[0.8], avoid_layers=[(1, 0)]
    )
    return c


if __name__ == "__main__":
    """FIXME, this fill looks weird"""
    m = mask(cache=False)
    m.show()
