import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler180
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.types import ComponentSpec, CrossSectionSpec, Optional


@gf.cell
def cutback_splitter(
    component: ComponentSpec = mmi1x2,
    cols: int = 4,
    rows: int = 5,
    port1: str = "o1",
    port2: str = "o2",
    port3: str = "o3",
    bend180: ComponentSpec = bend_euler180,
    straight: ComponentSpec = straight_function,
    mirror: bool = False,
    straight_length: Optional[float] = None,
    cross_section: CrossSectionSpec = "strip",
    **kwargs
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
    xs = gf.get_cross_section(cross_section, **kwargs)

    component = gf.get_component(component)
    bendu = gf.get_component(bend180, cross_section=xs)
    straight_component = gf.get_component(
        straight, length=straight_length or xs.radius * 2, cross_section=xs
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

    seq = component_sequence(sequence=s, symbol_to_component=symbol_to_component)

    c = gf.Component()
    ref = c << seq
    c.add_ports(ref.ports)

    n = len(s) - 2
    c.copy_child_info(component)
    c.info["components"] = n
    return c


if __name__ == "__main__":
    c = cutback_splitter()
    c.show(show_ports=True)
