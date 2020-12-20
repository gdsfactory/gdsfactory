"""One problem is that when we add references we have to make sure they have unique names.

The photonics package `pp` has a cell decorator that names the objects that it produces depending on the parameters that we pass them
"""


import pp


@pp.cell
def waveguide_cell(width=10, height=1):
    """Returns waveguide with automatic name."""
    wg = pp.Component("waveguide")
    wg.add_polygon([(0, 0), (width, 0), (width, height), (0, height)])
    wg.add_port(name="wgport1", midpoint=[0, height / 2], width=height, orientation=180)
    wg.add_port(
        name="wgport2", midpoint=[width, height / 2], width=height, orientation=0
    )
    return wg


if __name__ == "__main__":
    c = waveguide_cell()
    print(c)

    c = waveguide_cell(width=0.5)
    print(c)
