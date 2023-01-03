"""FIXME.

How can we differentiate the labels for two different references without changing the name?
At the moment we are using the labels
"""


if __name__ == "__main__":
    import gdsfactory as gf

    mzis = [gf.c.mzi()] * 2
    mzis_with_gc = [gf.routing.add_fiber_array(c, with_loopback=False) for c in mzis]
    c = gf.grid(mzis_with_gc)
    c.show()
