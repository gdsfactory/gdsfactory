from __future__ import annotations

from simphony.libraries.siepic import Waveguide

from gdsfactory.config import logger


def straight(
    length: float = 10.0,
    width: float = 0.5,
    thickness: float = 0.22,
    sw_angle: float = 90.0,
    **kwargs,
):
    """Return simphony Model for a Straight straight.

    Args:
        length: Length of the straight in um.
        width: Width of the straight in um (Valid for 0.4-0.6).
        thickness: Thickness of straight in um (Valid for 180nm-240nm).
        sw_angle: Sidewall angle. Valid for 80-90 degrees.
        kwargs: geometrical args that this model ignores

    """
    logger.info(f"ignoring {kwargs.keys()}")
    width *= 1e-6
    thickness *= 1e-6
    length *= 1e-6

    model = Waveguide(
        width=width, thickness=thickness, sw_angle=sw_angle, length=length
    )
    model.rename_pins("o1", "o2")
    return model


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony.plot_model import plot_model

    c = straight()

    plot_model(c, logscale=False)
    plt.show()
