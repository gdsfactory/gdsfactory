import numpy as np
from SiPANN.scee import Waveguide

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
        width: Width of the straights in microns.
        thickness: Thickness of the straights in microns.
        angle: Number of deg of circle that bent straight transverses.
        sw_angle: Sidewall angle from horizontal in degrees, ie 90 makes a square. Defaults to 90.
        kwargs: geometrical args that this model ignores.

    """
    c = gf.c.bend_euler(radius=radius, **kwargs)
    length = c.info["length"] * 1e3
    angle = np.deg2rad(angle)
    width *= 1e3
    thickness *= 1e3

    return Waveguide(width=width, thickness=thickness, sw_angle=sw_angle, length=length)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = bend_euler()
    wavelength = np.linspace(1500, 1600, 500)
    t = c.predict(wavelength)

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(wavelength, np.angle(t), label="t")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Phase (rad)")
    plt.title(r"Transmission at $\lambda \approx 1550nm$")
    plt.legend()
    plt.show()
