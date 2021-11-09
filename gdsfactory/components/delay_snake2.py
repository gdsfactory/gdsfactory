import warnings

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler180
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def delay_snake2(
    length: float = 1600.0,
    length0: float = 0.0,
    n: int = 2,
    bend180: ComponentFactory = bend_euler180,
    cross_section: CrossSectionFactory = strip,
    **kwargs,
) -> Component:
    """Snake input facing west
    Snake output facing east

    Args:
        length: total length
        length0: initial offset
        n: number of loops
        bend180
        cross_section: factory
        **kwargs: cross_section settings

    .. code::

       | length0 | length1 |

                 >---------|
                           |  bend180.length
       |-------------------|
       |
       |------------------->

       |   delta_length    |


    """
    if n % 2:
        warnings.warn(f"rounding {n} to {n//2 *2}", stacklevel=3)
        n = n // 2 * 2
    bend180 = bend180(cross_section=cross_section, **kwargs)
    delta_length = (length - length0 - n * bend180.info.length) / (n + 1)
    length1 = delta_length - length0
    assert (
        length1 > 0
    ), "Snake is too short: either reduce length0, increase the total length,\
    or decrease n"

    s1 = gf.components.straight(length=length1, cross_section=cross_section, **kwargs)
    sd = gf.components.straight(
        cross_section=cross_section, length=delta_length, **kwargs
    )

    symbol_to_component = {
        "_": (s1, "o1", "o2"),
        "-": (sd, "o1", "o2"),
        ")": (bend180, "o2", "o1"),
        "(": (bend180, "o1", "o2"),
    }

    sequence = "_)" + n // 2 * "-(-)"
    sequence = sequence[:-1]
    c = gf.components.component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )
    gf.port.auto_rename_ports(c)
    return c


def test_delay_snake2_length() -> Component:
    length = 200.0
    c = delay_snake2(n=2, length=length, layer=(2, 0))
    length_measured = (
        c.aliases[")1"].parent.info.length * 2 + c.aliases["-1"].parent.info.length * 3
    )
    assert np.isclose(
        length, length_measured
    ), f"length measured = {length_measured} != {length}"
    return c


if __name__ == "__main__":
    c = test_delay_snake2_length()
    c.show()
    # c = delay_snake2(n=2, length=200, layer=(2,0))
    # c.show()
