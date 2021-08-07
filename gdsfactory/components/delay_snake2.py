import warnings

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.tech import LIBRARY, Library
from gdsfactory.types import StrOrDict, StrOrDictOrNone


@gf.cell
def delay_snake2(
    length: float = 1600.0,
    length0: float = 0.0,
    n: int = 2,
    bend180: StrOrDictOrNone = None,
    waveguide: StrOrDict = "strip",
    library: Library = LIBRARY,
    **waveguide_settings,
) -> Component:
    """Snake input facing west
    Snake output facing east

    Args:
        length:
        length0: initial offset
        n: number of loops
        bend180
        waveguide
        library
        waveguide_settings

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
    bend180 = bend180 or dict(
        component="bend_euler", angle=180, waveguide=waveguide, **waveguide_settings
    )
    bend180 = library.get_component(bend180, waveguide=waveguide, **waveguide_settings)

    delta_length = (length - length0 - n * (bend180.length)) / (n + 1)
    length1 = delta_length - length0
    assert (
        length1 > 0
    ), "Snake is too short: either reduce length0, increase the total length,\
    or decrease n"

    s1 = gf.components.straight(
        waveguide=waveguide, length=length1, **waveguide_settings
    )
    sd = gf.components.straight(
        waveguide=waveguide, length=delta_length, **waveguide_settings
    )

    symbol_to_component = {
        "_": (s1, "W0", "E0"),
        "-": (sd, "W0", "E0"),
        ")": (bend180, "W1", "W0"),
        "(": (bend180, "W0", "W1"),
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
    c = delay_snake2(
        waveguide="strip",
        n=2,
        length=length,
        bend180=dict(component="bend_circular180"),
    )
    length_measured = (
        c.aliases[")1"].parent.length * 2 + c.aliases["-1"].parent.length * 3
    )
    assert np.isclose(
        length, length_measured
    ), f"length measured = {length_measured} != {length}"
    return c


if __name__ == "__main__":
    c = test_delay_snake2_length()
    c.show()
    # c = delay_snake2(waveguide="nitride", n=2, length=200)
    # c.show()
