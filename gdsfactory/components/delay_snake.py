from __future__ import annotations

import warnings

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler180
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

diagram = r"""
                 | length0   |

                 >---------\
                            \bend180.info['length']
                            /
       |-------------------/
       |
       |------------------->------->|
                            length2
       |   delta_length    |        |

"""


@gf.cell
def delay_snake(
    length: float = 1600.0,
    length0: float = 0.0,
    length2: float = 0.0,
    n: int = 2,
    bend180: ComponentSpec = bend_euler180,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    r"""Returns Snake with a starting bend and 180 bends.

    Args:
        length: total length.
        length0: start length.
        length2: end length.
        n: number of loops.
        bend180: ubend spec.
        cross_section: cross_section spec.
        kwargs: cross_section settings.

    .. code::

                 | length0   |

                 >---------\
                            \bend180.info['length']
                            /
       |-------------------/
       |
       |------------------->------->|
                            length2
       |   delta_length    |        |


    """
    if n % 2:
        warnings.warn(f"rounding {n} to {n//2 *2}", stacklevel=3)
        n = n // 2 * 2
    bend180 = gf.get_component(bend180, cross_section=cross_section, **kwargs)

    delta_length = (length - length0 - length2 - n * bend180.info["length"]) / n
    if delta_length < 0:
        raise ValueError(
            "Snake is too short: either reduce length0, length2, "
            f"increase the total length, or decrease the number of loops (n = {n}). "
            f"delta_length = {int(delta_length)}\n" + diagram
        )

    s0 = straight(cross_section=cross_section, length=length0, **kwargs)
    sd = straight(cross_section=cross_section, length=delta_length, **kwargs)
    s2 = straight(cross_section=cross_section, length=length2, **kwargs)

    symbol_to_component = {
        "_": (s0, "o1", "o2"),
        "-": (sd, "o1", "o2"),
        ")": (bend180, "o2", "o1"),
        "(": (bend180, "o1", "o2"),
        ".": (s2, "o1", "o2"),
    }

    sequence = "_)" + n // 2 * "-(-)"
    sequence = f"{sequence[:-1]}."
    return gf.components.component_sequence(
        sequence=sequence, symbol_to_component=symbol_to_component
    )


if __name__ == "__main__":
    # c = test_delay_snake3_length()

    length = 1562
    c = delay_snake(
        n=2,
        length=length,
        length2=length - 120,
        cross_section="strip",
    )
    # length_computed = c.area() / 0.5
    # assert np.isclose(length, length_computed), length_computed
    c.show()
