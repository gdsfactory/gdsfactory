import gdsfactory as gf
from gdsfactory.types import CrossSectionSpec
from pydantic import validate_arguments


@validate_arguments
def straight2a(
    length1: float = 5,
    length2: float = 10,
    cross_section1: CrossSectionSpec = gf.cross_section.strip,
    cross_section2: CrossSectionSpec = gf.cross_section.pin,
) -> gf.Component:
    """Returns a concatenation of two cross_sections.

    Args:
        length1: for the first section.
        length1: for the second section.
        cross_section1: for the input.
        cross_section2: for the output.
    """
    c = gf.Component()

    wg1 = gf.components.straight(length=length1, cross_section=cross_section1)
    wg2 = gf.components.straight(length=length2, cross_section=cross_section2)

    w1 = c.add_ref(wg1)
    w2 = c.add_ref(wg2)
    w2.connect("o1", w1.ports["o2"])
    return c


straight2b = gf.partial(
    straight2a,
    cross_section2=gf.cross_section.strip_heater_metal,
)


if __name__ == "__main__":
    c = straight2a()  # works
    # c = straight2b()  # FIXME: does not work
    c.show(show_ports=True)
