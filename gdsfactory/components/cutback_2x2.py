from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular180
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.mmi2x2 import mmi2x2
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Optional


@gf.cell
def bendu_double(
    component: ComponentSpec,
    cross_section: CrossSectionSpec = "strip",
    bend180: ComponentSpec = bend_circular180,
    port1: str = "o1",
    port2: str = "o2",
    port3: str = "o3",
    port4: str = "o4",
    **kwargs,
) -> ComponentSpec:
    """Returns double bend"""
    xs = gf.get_cross_section(cross_section, **kwargs)
    bendu = gf.Component()
    bend_r = bendu << bend180(cross_section=xs)
    bend_r2 = bendu << bend180(
        cross_section=xs,
        radius=xs.radius - (component.ports[port1].y - component.ports[port2].y),
    )
    bend_r2 = bend_r2.move(
        origin=(0, 0),
        destination=(0, component.ports[port1].y - component.ports[port2].y),
    )
    bendu.add_port("o1", port=bend_r.ports["o1"])
    bendu.add_port("o2", port=bend_r2.ports["o1"])
    bendu.add_port("o3", port=bend_r2.ports["o2"])
    bendu.add_port("o4", port=bend_r.ports["o2"])
    return bendu


@gf.cell
def straight_double(
    component: ComponentSpec,
    cross_section: CrossSectionSpec = "strip",
    port1: str = "o1",
    port2: str = "o2",
    straight_length: Optional[float] = None,
    **kwargs,
) -> ComponentSpec:
    """Returns double straight"""
    xs = gf.get_cross_section(cross_section, **kwargs)

    straight_double = gf.Component("straight_double")
    straight_component = straight(
        length=straight_length or xs.radius * 2, cross_section=xs
    )
    straight_component2 = straight(
        length=straight_length or xs.radius * 2, cross_section=xs
    )
    straight_r = straight_double << straight_component
    straight_r2 = straight_double << straight_component2.mirror((1, 0))
    straight_r2 = straight_r2.move(
        origin=(0, 0),
        destination=(0, -component.ports[port1].y + component.ports[port2].y),
    )
    straight_double.add_port("o1", port=straight_r.ports["o1"])
    straight_double.add_port("o2", port=straight_r2.ports["o1"])
    straight_double.add_port("o3", port=straight_r2.ports["o2"])
    straight_double.add_port("o4", port=straight_r.ports["o2"])
    return straight_double


@gf.cell
def cutback_2x2(
    component: ComponentSpec = mmi2x2,
    cols: int = 4,
    rows: int = 5,
    port1: str = "o1",
    port2: str = "o2",
    port3: str = "o3",
    port4: str = "o4",
    bend180: ComponentSpec = bend_circular180,
    mirror: bool = False,
    straight_length: Optional[float] = None,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns a daisy chain of splitters for measuring their loss.

    Args:
        component: for cutback.
        cols: number of columns.
        rows: number of rows.
        port1: name of first optical port.
        port2: name of second optical port.
        bend180: ubend.
        straight: waveguide spec to connect both sides.
        mirror: Flips component. Useful when 'o2' is the port that you want to route to.
        straight_length: length of the straight section between cutbacks.
        cross_section: specification (CrossSection, string or dict).
        kwargs: cross_section settings.
    """
    component = gf.get_component(component)

    bendu = bendu_double(
        component=component,
        cross_section=cross_section,
        bend180=bend180,
        port1=port1,
        port2=port2,
        port3=port3,
        port4=port4,
        **kwargs,
    )

    straight = straight_double(
        component=component,
        cross_section=cross_section,
        straight_length=straight_length,
        port1=port1,
        port2=port2,
        **kwargs,
    )

    # Define a map between symbols and (component, input port, output port)
    symbol_to_component = {
        "A": (component, port1, port3),
        "B": (component, port4, port2),
        "D": (bendu, "o2", "o3"),
        "C": (bendu, "o4", "o1"),
        "-": (straight, "o1", "o3"),
        "_": (straight, "o2", "o4"),
    }

    # Generate the sequence of staircases
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
    n = cols * rows * 2

    seq = component_sequence(sequence=s, symbol_to_component=symbol_to_component)

    c = gf.Component()
    c << seq
    c.add_port("o1", port=seq.named_references["A1"]["o1"])
    c.add_port("o2", port=seq.named_references["A1"]["o2"])
    c.add_port("o3", port=seq.named_references[f"B{n}"]["o2"])
    c.add_port("o4", port=seq.named_references[f"B{n}"]["o1"])

    c.copy_child_info(component)
    c.info["components"] = 2 * n
    return c


if __name__ == "__main__":
    c = cutback_2x2(cols=3, rows=2, mirror=True)
    c.show(show_ports=True)
