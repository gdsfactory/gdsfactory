"""
FIXME: this file generates DRC error non-orientable boundary

- define cross_section extrusion for particular components, such as bends

"""

import gdsfactory as gf


if __name__ == "__main__":

    c = gf.c.bend_circular(cross_section=gf.cross_section.pin, radius=5)
    c.show()
