from numpy import float64

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular, bend_circular180
from gdsfactory.components.bend_euler import bend_euler, bend_euler180
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.straight import straight as straight_function
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
    rows: int = 6,
    columns: int = 5,
    straight: ComponentFactory = straight_function,
    **kwargs
):
    """Deprecated! use cutback_bend90 instead,
    which has smaller footprint

    Args:
        bend90:
        straight_length:
        rows:
        columns:
        straight: function for straight

    keyword args:
        cross_section:


    .. code::

        this is a column
            _
          _|
        _|

        _ this is a row

    """

    bend90 = gf.call_if_func(bend90)
    straightx = straight(
        length=straight_length, width=bend90.ports["o1"].width, **kwargs
    )

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend90, "o1", "o2"),
        "B": (bend90, "o2", "o1"),
        "S": (straightx, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(columns):
        s += "ASBS" * rows
        s += "ASAS" if i % 2 == 0 else "BSBS"
    s = s[:-4]

    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=90
    )
    c.info.n_bends = rows * columns * 2 + columns * 2 - 2
    return c


@cell
def cutback_bend90(
    bend90: ComponentOrFactory = bend_euler,
    straight_length: float = 5.0,
    rows: int = 6,
    columns: int = 6,
    spacing: int = 5,
    straight: ComponentFactory = straight_function,
) -> Component:
    """

    .. code::

           _
        |_| |

    """
    bend90 = gf.call_if_func(bend90)
    straightx = straight(length=straight_length, width=bend90.ports["o1"].width)

    straight_length = 2 * _get_bend_size(bend90) + spacing + straight_length
    straighty = straight(
        length=straight_length,
        width=bend90.ports["o1"].width,
    )
    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend90, "o1", "o2"),
        "B": (bend90, "o2", "o1"),
        "-": (straightx, "o1", "o2"),
        "|": (straighty, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(columns):
        if i % 2 == 0:  # even row
            s += "A-A-B-B-" * rows + "|"
        else:
            s += "B-B-A-A-" * rows + "|"
    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=0
    )
    c.info.n_bends = rows * columns * 4
    return c


@cell
def staircase(
    bend90: ComponentOrFactory = bend_euler,
    length_v: float = 5.0,
    length_h: float = 5.0,
    rows: int = 4,
    straight: ComponentFactory = straight_function,
) -> Component:
    bend90 = gf.call_if_func(bend90)

    wgh = straight(length=length_h, width=bend90.ports["o1"].width)
    wgv = straight(length=length_v, width=bend90.ports["o1"].width)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend90, "o1", "o2"),
        "B": (bend90, "o2", "o1"),
        "-": (wgh, "o1", "o2"),
        "|": (wgv, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = "-A|B" * rows + "-"

    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=0
    )
    c.info.n_bends = 2 * rows
    return c


@cell
def cutback_bend180(
    bend180: ComponentOrFactory = bend_euler180,
    straight_length: float = 5.0,
    rows: int = 6,
    columns: int = 6,
    spacing: int = 3,
    straight: ComponentFactory = straight_function,
) -> Component:
    """

    .. code::

          _
        _| |_  this is a row

        _ this is a column

    """
    bend180 = gf.call_if_func(bend180)

    straightx = straight(length=straight_length, width=bend180.ports["o1"].width)
    wg_vertical = straight(
        length=2 * bend180.size_info.width + straight_length + spacing,
        width=bend180.ports["o1"].width,
    )

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "D": (bend180, "o1", "o2"),
        "C": (bend180, "o2", "o1"),
        "-": (straightx, "o1", "o2"),
        "|": (wg_vertical, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(columns):
        if i % 2 == 0:  # even row
            s += "D-C-" * rows + "|"
        else:
            s += "C-D-" * rows + "|"
    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=0
    )
    c.info.n_bends = rows * columns * 2 + columns * 2 - 2
    return c


cutback_bend180circular = gf.partial(cutback_bend180, bend180=bend_circular180)
cutback_bend90circular = gf.partial(cutback_bend90, bend90=bend_circular)

if __name__ == "__main__":
    c = cutback_bend()
    # c = cutback_bend90()
    # c = cutback_bend_circular(rows=7, columns=4, radius=5) #62
    # c = cutback_bend_circular(rows=14, columns=4) #118
    # c = cutback_bend90()
    # c = cutback_bend180(rows=3, columns=1)
    # c = cutback_bend(rows=3, columns=2)
    # c = cutback_bend90(rows=3, columns=2)
    # c = cutback_bend180(rows=2, columns=2)
    # c = cutback_bend(rows=3, columns=2)
    c.show()
