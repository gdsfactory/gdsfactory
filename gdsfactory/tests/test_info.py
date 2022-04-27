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


def test_args() -> None:
    c1 = gf.components.pad((150, 150))
    assert c1.settings.full["size"][0] == 150


if __name__ == "__main__":
    test_args()
    # assert c1.settings.['full']['size'][0] == 150
    # c1 = gf.components.pad((150, 150))
    # c1.show()
