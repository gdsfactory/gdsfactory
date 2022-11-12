"""gdstk does not allow you to create the same cell name twice on the library."""

import gdsfactory as gf

if __name__ == "__main__":
    c1 = gf.Component("a")
    c2 = gf.Component("a")
