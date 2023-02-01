from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gdsdiff.gdsdiff import gdsdiff

if __name__ == "__main__":
    c1 = gf.components.mmi1x2(length_mmi=5)
    c2 = gf.components.mmi1x2(length_mmi=9)
    c3 = gdsdiff(c1, c2)
    gf.show(c3)
