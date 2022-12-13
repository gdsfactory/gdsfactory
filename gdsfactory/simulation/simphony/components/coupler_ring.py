from __future__ import annotations

from simphony.libraries import sipann


def coupler_ring(
    radius: float = 5.0,
    width: float = 0.5,
    thickness: float = 0.22,
    gap: float = 0.22,
    length_x: float = 4.0,
    sw_angle: float = 90.0,
    **kwargs,
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

           o2 \           / o3
               \         /
                ---------
           o1---------------o4
                 length_x

    .. plot::
        :include-source:

        import gdsfactory as gf
        c = gf.components.coupler_ring()
        c.plot()


    .. plot::
        :include-source:

        import gdsfactory.simulation.simphony as gs
        import gdsfactory.simulation.simphony.components as gc

        m = gc.coupler_ring()
        gs.plot_model(m)

    """
    width *= 1e-6
    thickness *= 1e-6
    gap *= 1e-6
    length = length_x * 1e-6
    radius *= 1e-6

    model = sipann.HalfRacetrack(
        radius=radius,
        width=width,
        thickness=thickness,
        gap=gap,
        length=length,
        sw_angle=sw_angle,
    )
    model.rename_pins("o1", "o2", "o4", "o3")

    return model


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from gdsfactory.simulation.simphony.plot_model import plot_model

    c = coupler_ring()
    print(c)
    wavelengths = np.linspace(1.5, 1.6) * 1e-6
    plot_model(c, wavelengths=wavelengths)
    plt.show()
