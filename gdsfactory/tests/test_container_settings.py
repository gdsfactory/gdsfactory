"""
info:

- default
- changed
- full
- info (derived properties)
- child: if any

Calculated/derived properties are stored in info

"""

import gdsfactory as gf


def test_args():
    c1 = gf.c.pad((150, 150))
    assert c1.info.full.size[0] == 150


if __name__ == "__main__":
    test_args()
    # assert c1.settings.size.full[0] == 150
    c1 = gf.c.pad((150, 150))
    c2 = gf.add_padding(c1)
    c2.show()
    c3 = gf.add_padding(c2)
