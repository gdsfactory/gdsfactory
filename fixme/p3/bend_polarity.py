"""
FIXME,

- bends in spirals flip the polarity
- if cross_section is wider than bend radius, it produces non-orientable boundaries

"""

import gdsfactory as gf


if __name__ == "__main__":

    c = gf.components.spiral_inner_io(
        cross_section=gf.cross_section.pin,
        radius=5,
        contact_width=2,
        waveguide_spacing=5,
    )
    c.show()
