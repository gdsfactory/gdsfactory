"""
Not sure how to handle this case
"""

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.components.spiral(direction="NORTH")
    c.show()
    cc = gf.routing.add_fiber_single(component=c)
    cc.show()
