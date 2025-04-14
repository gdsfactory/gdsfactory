from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.containers.component_sequence import component_sequence
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
def cutback_splitter(
    component: ComponentSpec = "mmi1x2",
    cols: int = 4,
    rows: int = 5,
    port1: str = "o1",
    port2: str = "o2",
    port3: str = "o3",
    bend180: ComponentSpec = "bend_euler180",
    mirror: bool = False,
    straight: ComponentSpec = "straight",
    straight_length: float | None = None,
    cross_section: CrossSectionSpec = "strip",
    **kwargs: Any,
) -> Component:
    """Returns a daisy chain of splitters for measuring their loss.

    Args:
        component: for cutback.
        cols: number of columns.
        rows: number of rows.
        port1: name of first optical port.
        port2: name of second optical port.
        port3: name of third optical port.
        bend180: ubend.
        mirror: Flips component. Useful when 'o2' is the port that you want to route to.
        straight: waveguide spec to connect both sides.
        straight_length: length of the straight section between cutbacks.
        cross_section: specification (CrossSection, string or dict).
        kwargs: cross_section settings.
    """
    xs = gf.get_cross_section(cross_section, **kwargs)

    component = gf.get_component(component)
    bendu = gf.get_component(bend180, cross_section=xs)
    radius = xs.radius
    assert radius is not None
    straight_component = gf.get_component(
        straight,
        length=straight_length or radius * 2,
        cross_section=xs,
    )

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (component, port1, port2),
        "B": (component, port3, port1),
        "D": (bendu, "o1", "o2"),
        "C": (bendu, "o2", "o1"),
        "-": (straight_component, "o1", "o2"),
        "_": (straight_component, "o2", "o1"),
    }

    s = ""
    for i in range(rows):
        s += "AB" * cols
        if mirror:
            s += "C" if i % 2 == 0 else "D"
        else:
            s += "D" if i % 2 == 0 else "C"

    s = s[:-1]
    s += "-_"

    for i in range(rows):
        s += "AB" * cols
        s += "D" if (i + rows) % 2 == 0 else "C"

    s = s[:-1]

    c = component_sequence(sequence=s, symbol_to_component=symbol_to_component)
    n = len(s) - 2
    c.info["components"] = n
    return c


if __name__ == "__main__":
    c = cutback_splitter()
    c.show()
