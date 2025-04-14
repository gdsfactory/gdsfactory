from __future__ import annotations

from functools import partial
from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.containers.component_sequence import component_sequence
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
def cutback_component(
    component: ComponentSpec = "taper_0p5_to_3_l36",
    cols: int = 4,
    rows: int = 5,
    port1: str = "o1",
    port2: str = "o2",
    bend180: ComponentSpec = "bend_euler180",
    mirror: bool = False,
    mirror1: bool = False,
    mirror2: bool = False,
    straight_length: float | None = None,
    straight_length_pair: float | None = None,
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    **kwargs: Any,
) -> Component:
    """Returns a daisy chain of components for measuring their loss.

    Works only for components with 2 ports (input, output).

    Args:
        component: for cutback.
        cols: number of columns.
        rows: number of rows.
        port1: name of first optical port.
        port2: name of second optical port.
        bend180: ubend.
        mirror: Flips component. Useful when 'o2' is the port that you want to route to.
        mirror1: mirrors first component.
        mirror2: mirrors second component.
        straight_length: length of the straight section between cutbacks.
        straight_length_pair: length of the straight section between each component pair.
        cross_section: specification (CrossSection, string or dict).
        straight: straight spec.
        kwargs: component settings.
    """
    xs = gf.get_cross_section(cross_section)

    component = gf.get_component(component, **kwargs)
    bendu = gf.get_component(bend180, cross_section=xs)

    radius = xs.radius
    assert radius is not None
    straight_length = radius * 2 if straight_length is None else straight_length
    straight_component = gf.get_component(
        straight, length=straight_length, cross_section=xs
    )
    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (component, port1, port2),
        "B": (component, port2, port1),
        "D": (bendu, "o1", "o2"),
        "C": (bendu, "o2", "o1"),
        "-": (straight_component, "o1", "o2"),
        "_": (straight_component, "o2", "o1"),
    }
    if straight_length_pair:
        straight_pair = gf.get_component(
            straight, length=straight_length_pair, cross_section=xs
        )
        symbol_to_component["."] = (straight_pair, "o2", "o1")

    # Generate the sequence of staircases
    s = ""
    for i in range(rows):
        a = "!A" if mirror1 else "A"
        b = "!B" if mirror2 else "B"

        if straight_length_pair:
            s += f"{a}.{b}" * cols if straight_length_pair else (a + b) * cols
        else:
            s += f"{a}{b}" * cols if straight_length_pair else (a + b) * cols

        if mirror:
            s += "C" if i % 2 == 0 else "D"
        else:
            s += "D" if i % 2 == 0 else "C"

    s = s[:-1]
    s += "-_"

    for i in range(rows):
        if straight_length_pair:
            s += f"{a}.{b}" * cols if straight_length_pair else (a + b) * cols
        else:
            s += f"{a}{b}" * cols if straight_length_pair else (a + b) * cols
        s += "D" if (i + rows) % 2 == 0 else "C"

    s = s[:-1]

    c = component_sequence(sequence=s, symbol_to_component=symbol_to_component)
    n = 2 * s.count("A")
    c.info["components"] = n
    return c


cutback_component_mirror = partial(cutback_component, mirror=True)

if __name__ == "__main__":
    c = cutback_component_mirror()
    # c = cutback_component_mirror(component=component_flipped)
    # c = gf.routing.add_fiber_single(c)
    c.show()
