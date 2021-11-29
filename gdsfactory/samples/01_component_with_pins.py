"""You add pins in a pin layer to clearly see the component ports."""

import gdsfactory as gf


@gf.cell
def straight_sample(length=5, width=1):
    """Returns straight with ports."""
    wg = gf.Component("straight_sample")
    wg.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=(1, 0))
    wg.add_port(name="o1", midpoint=[0, width / 2], width=width, orientation=180)
    wg.add_port(name="o2", midpoint=[length, width / 2], width=width, orientation=0)
    return wg


if __name__ == "__main__":
    wg = straight_sample()
    gf.add_pins(wg)
    wg.show()
