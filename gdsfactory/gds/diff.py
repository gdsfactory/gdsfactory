import gdsfactory
from gdsfactory.gdsdiff.gdsdiff import gdsdiff

if __name__ == "__main__":
    c1 = gdsfactory.components.mmi1x2(length_mmi=5)
    c2 = gdsfactory.components.mmi1x2(length_mmi=9)
    c3 = gdsdiff(c1, c2)
    gdsfactory.show(c3)
