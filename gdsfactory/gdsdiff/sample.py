import gdsfactory as gf
from gdsfactory.gdsdiff.gdsdiff import gdsdiff

if __name__ == "__main__":

    c1 = gf.c.straight(length=2)
    c2 = gf.c.straight(length=3)
    c = gdsdiff(c1, c2)
    c.show()
