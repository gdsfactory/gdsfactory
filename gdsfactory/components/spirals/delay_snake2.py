from __future__ import annotations

import warnings

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.containers.component_sequence import component_sequence
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

diagram = """
       | length0 | length1 |

                 >---------|
                           | bend180.length
       |-------------------|
       |
       |------------------->------- |
                            length2
       |   delta_length    |        |
"""


@gf.cell
def delay_snake2(
    length: float = 1600.0,
    length0: float = 0.0,
    length2: float = 0.0,
    n: int = 2,
    bend180: ComponentSpec = "bend_euler180",
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
) -> Component:
    """Returns Snake with a starting straight and 180 bends.

    Input faces west output faces east.

    Args:
        length: total length.
        length0: start length.
        length2: end length.
        n: number of loops.
        bend180: ubend spec.
        cross_section: cross_section spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.

    .. code::

       | length0 | length1 |

                 >---------|
                           | bend180.length
       |-------------------|
       |
       |------------------->------- |
                            length2
       |   delta_length    |        |
    """
    if n % 2:
        warnings.warn(f"rounding {n} to {n // 2 * 2}", stacklevel=3)
        n = n // 2 * 2

    bend180 = gf.get_component(bend180, cross_section=cross_section, width=width)

    delta_length = (length - length0 - length2 - n * bend180.info["length"]) / (n + 1)
    length1 = delta_length - length0
    if length1 < 0:
        raise ValueError(
            "Snake is too short: either reduce length0, length2, "
            f"increase the total length, or decrease the number of loops (n = {n}). "
            f"length1 = {int(length1)}, delta_length = {int(delta_length)}\n" + diagram
        )

    s1 = gf.components.straight(
        length=length1, cross_section=cross_section, width=width
    )
    s2 = gf.components.straight(
        length=length2, cross_section=cross_section, width=width
    )
    sd = gf.components.straight(
        cross_section=cross_section, length=delta_length, width=width
    )

    symbol_to_component = {
        "_": (s1, "o1", "o2"),
        "-": (sd, "o1", "o2"),
        ")": (bend180, "o2", "o1"),
        "(": (bend180, "o1", "o2"),
        ".": (s2, "o1", "o2"),
    }

    sequence = "_)" + n // 2 * "-(-)"
    sequence = sequence[:-1]
    sequence += "."
    return component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )


if __name__ == "__main__":
    import gdsfactory as gf

    # test_length_delay_snake2()
    # c.show( )
    # c = delay_snake2(n=2, length=500, layer=(2, 0), length0=100)
    c = delay_snake2()
    # c = gf.grid((gf.c.delay_snake, delay_snake2(length0=100), gf.c.delay_snake3))
    c.show()
