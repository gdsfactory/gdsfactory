from __future__ import annotations

import warnings

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def delay_snake2(
    length: float = 1600.0,
    length0: float = 0.0,
    n: int = 2,
    bend180: ComponentSpec = "bend_euler180",
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns Snake with a starting straight and 180 bends.

    Input faces west output faces east.

    Args:
        length: total length.
        length0: initial offset.
        n: number of loops.
        bend180: ubend spec.
        cross_section: cross_section spec.
        kwargs: cross_section settings.

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

    bend180 = gf.get_component(bend180, cross_section=cross_section, **kwargs)

    delta_length = (length - length0 - n * bend180.info["length"]) / (n + 1)
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
    return gf.components.component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )


def test_length_delay_snake2():
    import numpy as np

    length = 200.0
    c = delay_snake2(length=length, cross_section="strip_no_pins")
    length_computed = c.area() / 0.5
    np.isclose(length, length_computed)
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    # c = test_length_delay_snake2()
    # c.show(show_ports=True)
    # c = delay_snake2(n=2, length=500, layer=(2, 0), length0=100)
    # c = delay_snake2()
    c = gf.grid([gf.c.delay_snake, delay_snake2(length0=100), gf.c.delay_snake3])
    c.show(show_ports=True)
