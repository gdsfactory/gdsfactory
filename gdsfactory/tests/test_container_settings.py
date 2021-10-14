"""
settings:

- default
- changed
- full
- info (derived properties)

Calculated/derived properties are stored in info

arguments: input settings
settings:
parameters:

info: derived settings

inputs:
outputs:


settings_input
setting_derived


Sbend:
    - width and height
    - or radius


get_settings
info:

"""

import gdsfactory as gf


def test_args():
    c1 = gf.c.pad((150, 150))
    assert c1.settings["size"][0] == 150


if __name__ == "__main__":
    c1 = gf.c.pad((150, 150))
    c2 = gf.add_padding_container(c1)
