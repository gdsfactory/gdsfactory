from __future__ import annotations

from functools import partial
from itertools import islice
from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.containers.component_sequence import component_sequence
from gdsfactory.typings import ComponentSpec


def _get_bend_size(bend90: Component) -> float:
    # Use islice to efficiently fetch first 2 ports from bend90.ports, avoiding list creation
    p1, p2 = islice(bend90.ports, 2)
    dx = abs(p2.x - p1.x)
    dy = abs(p2.y - p1.y)
    return max(dx, dy)


@gf.cell_with_module_name
def cutback_bend(
    component: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    straight_length: float = 5.0,
    rows: int = 6,
    cols: int = 5,
    **kwargs: Any,
) -> Component:
    """We recommend using cutback_bend90 instead for a smaller footprint.

    Args:
        component: bend spec.
        straight: straight spec.
        straight_length: in um.
        rows: number of rows.
        cols: number of cols.
        kwargs: cross_section settings.

    .. code::

        this is a column
            _
          _|
        _|

        _ this is a row
    """
    from gdsfactory.pdk import get_component

    bend90 = get_component(component, **kwargs)
    straightx = gf.get_component(straight, length=straight_length, **kwargs)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend90, "o1", "o2"),
        "B": (bend90, "o2", "o1"),
        "S": (straightx, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(cols):
        s += "ASBS" * rows
        s += "ASAS" if i % 2 == 0 else "BSBS"
    s = s[:-4]

    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=90
    )
    c.info["components"] = rows * cols * 2 + cols * 2 - 2
    return c


@gf.cell_with_module_name
def cutback_bend90(
    component: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    straight_length: float = 5.0,
    rows: int = 6,
    cols: int = 6,
    spacing: int = 5,
    **kwargs: Any,
) -> Component:
    """Returns bend90 cutback.

    Args:
        component: bend spec.
        straight: straight spec.
        straight_length: in um.
        rows: number of rows.
        cols: number of cols.
        spacing: in um.
        kwargs: cross_section settings.

    .. code::

           _
        |_| |
    """
    bend90 = gf.get_component(component, **kwargs)
    straightx = gf.get_component(straight, length=straight_length, **kwargs)
    straight_length = 2 * _get_bend_size(bend90) + spacing + straight_length
    straighty = gf.get_component(straight, length=straight_length, **kwargs)

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (bend90, "o1", "o2"),
        "B": (bend90, "o2", "o1"),
        "-": (straightx, "o1", "o2"),
        "|": (straighty, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = "".join(
        "A-A-B-B-" * rows + "|" if i % 2 == 0 else "B-B-A-A-" * rows + "|"
        for i in range(cols)
    )
    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=0
    )
    c.info["components"] = rows * cols * 4
    return c


@gf.cell_with_module_name
def staircase(
    component: ComponentSpec | Component = "bend_euler",
    straight: ComponentSpec = "straight",
    length_v: float = 5.0,
    length_h: float = 5.0,
    rows: int = 4,
    **kwargs: Any,
) -> Component:
    """Returns staircase.

    Args:
        component: bend spec.
        straight: straight spec.
        length_v: vertical length.
        length_h: vertical length.
        rows: number of rows.
        cols: number of cols.
        kwargs: cross_section settings.
    """
    bend90 = (
        component
        if isinstance(component, Component)
        else gf.get_component(component, **kwargs)
    )

    wgh = gf.get_component(straight, length=length_h, **kwargs)
    wgv = gf.get_component(straight, length=length_v, **kwargs)

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
        sequence=s,
        symbol_to_component=symbol_to_component,
        start_orientation=0,
    )
    c.info["components"] = 2 * rows
    return c


@gf.cell_with_module_name
def cutback_bend180(
    component: ComponentSpec = "bend_euler180",
    straight: ComponentSpec = "straight",
    straight_length: float = 5.0,
    rows: int = 6,
    cols: int = 6,
    spacing: float = 3.0,
    **kwargs: Any,
) -> Component:
    """Returns cutback to measure u bend loss.

    Args:
        component: bend spec.
        straight: straight spec.
        straight_length: in um.
        rows: number of rows.
        cols: number of cols.
        spacing: in um.
        kwargs: cross_section settings.

    .. code::

          _
        _| |_  this is a row

        _ this is a column
    """
    bend180 = gf.get_component(component, **kwargs)
    straightx = gf.get_component(straight, length=straight_length, **kwargs)
    wg_vertical = gf.get_component(
        straight,
        length=2 * bend180.xsize + straight_length + spacing,
        **kwargs,
    )

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "D": (bend180, "o1", "o2"),
        "C": (bend180, "o2", "o1"),
        "-": (straightx, "o1", "o2"),
        "|": (wg_vertical, "o1", "o2"),
    }

    # Generate the sequence of staircases
    s = "".join(
        "D-C-" * rows + "|" if i % 2 == 0 else "C-D-" * rows + "|" for i in range(cols)
    )

    s = s[:-1]

    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=0
    )
    c.info["components"] = rows * cols * 2 + cols * 2 - 2
    return c


cutback_bend180circular = partial(cutback_bend180, component="bend_circular180")
cutback_bend90circular = partial(cutback_bend90, component="bend_circular")

if __name__ == "__main__":
    # c = cutback_bend()
    c = cutback_bend90()
    # c = cutback_bend90circular(rows=7, cols=4)
    # c = cutback_bend_circular(rows=14, cols=4)
    # c = cutback_bend90()
    # c = cutback_bend180(rows=3, cols=1)
    # c = cutback_bend(rows=3, cols=2)
    # c = cutback_bend90(rows=3, cols=2)
    # c = cutback_bend180(rows=2, cols=2)
    # c = cutback_bend(rows=3, cols=2)
    c.show()
