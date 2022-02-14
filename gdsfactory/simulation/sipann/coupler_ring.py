from SiPANN.scee import HalfRacetrack


def coupler_ring(
    radius: float = 5.0,
    width: float = 0.5,
    thickness: float = 0.22,
    gap: float = 0.22,
    length_x: float = 4.0,
    sw_angle: float = 90.0,
    **kwargs
):
    r"""Return model for for half a ring coupler.

    Args:
        radius: 5
        width: width um (Valid for 0.4-0.6)
        thickness: Thickness in um (Valid for 0.18-0.24)
        gap: distance between straights in um. (Must be > 0.1)
        length_x: Length of straight portion of coupler in um
        sw_angle: Sidewall angle from horizontal in degrees
        kwargs: geometrical args that this model ignores

    .. code::

           N0            N1
           |             |
            \           /
             \         /
           ---=========---
        W0    length_x    E0


            2 \           / 4
               \         /
                ---------
            1---------------3

    .. plot::
        :include-source:

        import gdsfactory as gf

        c = gf.components.coupler_ring()
        c.plot()


    .. plot::
        :include-source:

        import gdslib as gl

        m = gl.components.coupler_ring()
        gl.plot_model(m)

    """

    width = width * 1e3
    thickness = thickness * 1e3
    gap = gap * 1e3
    length = length_x * 1e3
    radius = radius * 1e3

    s = HalfRacetrack(
        radius=radius,
        width=width,
        thickness=thickness,
        gap=gap,
        length=length,
        sw_angle=sw_angle,
    )
    return s


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    import gdslib as gl

    c = coupler_ring()
    print(c)
    wavelengths = np.linspace(1.5, 1.6) * 1e-6
    gl.simphony.plot_model(c, wavelengths=wavelengths)
    plt.show()
