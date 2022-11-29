from __future__ import annotations

import numpy as np
from simphony.libraries.sipann import Waveguide

import gdsfactory as gf


def bend_euler(
    radius: float = 10.0,
    width: float = 0.5,
    thickness: float = 0.22,
    angle: float = 90,
    sw_angle: float = 90.0,
    **kwargs,
):
    """Return simphony Model for a bend using a straight.

    FIXME. this is fake bend! need to create a better model

    Args:
        radius: Radius of straight in microns.
        width: Width of the straights in microns
        thickness: Thickness of the straights in microns
        angle: Number of deg of circle that bent straight transverses
        sw_angle: Sidewall angle from horizontal in degrees, ie 90 makes a square. Defaults to 90.
        kwargs: geometrical args that this model ignores

    """
    c = gf.c.bend_euler(radius=radius, angle=angle)
    length = c.info["length"] * 1e-6
    angle = np.deg2rad(angle)
    width *= 1e-6
    thickness *= 1e-6

    model = Waveguide(
        width=width, thickness=thickness, sw_angle=sw_angle, length=length
    )
    model.rename_pins("o1", "o2")
    return model


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony.plot_model import plot_model

    c = bend_euler()
    wavelengths = np.linspace(1.5, 1.6) * 1e-6
    plot_model(c, wavelengths=wavelengths)
    plt.show()
