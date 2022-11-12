import numpy as np
from SiPANN.scee import Waveguide


def bend_circular(
    radius: float = 10.0,
    width: float = 0.5,
    thickness: float = 0.22,
    angle: float = 90.0,
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
        sw_angle: Sidewall angle from horizontal in degrees. Defaults to 90.
        kwargs: geometrical args that this model ignores.

    """
    angle = np.deg2rad(angle)
    width *= 1e3
    thickness *= 1e3
    length = angle * radius * 1e3

    return Waveguide(width=width, thickness=thickness, sw_angle=sw_angle, length=length)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = bend_circular()

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
