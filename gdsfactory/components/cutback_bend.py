from numpy import float64

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular, bend_circular180
from gdsfactory.components.bend_euler import bend_euler, bend_euler180
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.straight import straight
from gdsfactory.types import ComponentFactory, ComponentOrFactory


def _get_bend_size(bend90: Component) -> float64:
    p1, p2 = list(bend90.ports.values())[:2]
    bsx = abs(p2.x - p1.x)
    bsy = abs(p2.y - p1.y)
    return max(bsx, bsy)


@cell
def cutback_bend(
    bend90: ComponentOrFactory = bend_euler,
    straight_length: float = 5.0,
    n_steps: int = 6,
    n_stairs: int = 5,
):
    """Deprecated! use cutback_bend90 instead!
    this is a stair

    .. code::
            _
          _|
        _|

        _ this is a step

    """

    bend90 = gf.call_if_func(bend90)
    wg = straight(length=straight_length, width=bend90.ports["o1"].width)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend90, "o1", "o2"),
        "B": (bend90, "o2", "o1"),
        "S": (wg, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(n_stairs):
        s += "ASBS" * n_steps
        s += "ASAS" if i % 2 == 0 else "BSBS"
    s = s[:-4]

    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=90
    )
    c.update_settings(n_bends=n_steps * n_stairs * 2 + n_stairs * 2 - 2)
    return c


@cell
def cutback_bend90(
    bend90: ComponentOrFactory = bend_euler,
    straight_length: float = 5.0,
    n_steps: int = 6,
    cols: int = 6,
    spacing: int = 5,
    wg_loop_length: None = None,
    straight_factory: ComponentFactory = straight,
) -> Component:
    """

    .. code::

           _
        |_| |

    """
    bend90 = gf.call_if_func(bend90)

    wg = straight_factory(length=straight_length, width=bend90.ports["o1"].width)
    if wg_loop_length is None:
        wg_loop_length = 2 * _get_bend_size(bend90) + spacing + straight_length

    wg_loop = straight_factory(
        length=wg_loop_length,
        width=bend90.ports["o1"].width,
    )
    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend90, "o1", "o2"),
        "B": (bend90, "o2", "o1"),
        "-": (wg, "o1", "o2"),
        "|": (wg_loop, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(cols):
        if i % 2 == 0:  # even row
            s += "A-A-B-B-" * n_steps + "|"
        else:
            s += "B-B-A-A-" * n_steps + "|"
    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=0
    )
    c.update_settings(n_bends=n_steps * cols * 4)
    return c


@cell
def staircase(
    bend90: ComponentOrFactory = bend_euler,
    length_v: float = 5.0,
    length_h: float = 5.0,
    n_steps: int = 4,
    straight_factory: ComponentFactory = straight,
) -> Component:
    bend90 = gf.call_if_func(bend90)

    wgh = straight_factory(length=length_h, width=bend90.ports["o1"].width)
    wgv = straight_factory(length=length_v, width=bend90.ports["o1"].width)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend90, "o1", "o2"),
        "B": (bend90, "o2", "o1"),
        "-": (wgh, "o1", "o2"),
        "|": (wgv, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = "-A|B" * n_steps + "-"

    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=0
    )
    c.update_settings(n_bends=2 * n_steps)
    return c


@cell
def cutback_bend180(
    bend180: ComponentOrFactory = bend_euler180,
    straight_length: float = 5.0,
    n_steps: int = 6,
    cols: int = 6,
    spacing: int = 3,
    straight_factory: ComponentFactory = straight,
) -> Component:
    """

    .. code::

          _
        _| |_| this is a stair

        _ this is a step

    """
    bend180 = gf.call_if_func(bend180)

    wg = straight_factory(length=straight_length, width=bend180.ports["o1"].width)
    wg_vertical = straight_factory(
        length=2 * bend180.size_info.width + straight_length + spacing,
        width=bend180.ports["o1"].width,
    )

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "D": (bend180, "o1", "o2"),
        "C": (bend180, "o2", "o1"),
        "-": (wg, "o1", "o2"),
        "|": (wg_vertical, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(cols):
        if i % 2 == 0:  # even row
            s += "D-C-" * n_steps + "|"
        else:
            s += "C-D-" * n_steps + "|"
    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=0
    )
    c.update_settings(n_bends=n_steps * cols * 2 + cols * 2 - 2)
    return c


cutback_bend180circular = gf.partial(cutback_bend180, bend180=bend_circular180)
cutback_bend90circular = gf.partial(cutback_bend90, bend90=bend_circular)

if __name__ == "__main__":
    # c = cutback_bend_circular(n_steps=7, n_stairs=4, radius=5) #62
    # c = cutback_bend_circular(n_steps=14, n_stairs=4) #118
    # c = cutback_bend90()
    c = cutback_bend180()
    c.show()
