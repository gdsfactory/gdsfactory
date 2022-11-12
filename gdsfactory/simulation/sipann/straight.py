from SiPANN.scee import Waveguide


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
        kwargs: geometrical args that this model ignores.

    """
    width *= 1e3
    thickness *= 1e3
    length *= 1e3

    return Waveguide(width=width, thickness=thickness, sw_angle=sw_angle, length=length)


pass
