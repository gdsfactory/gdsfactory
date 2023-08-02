from __future__ import annotations

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.typings import ComponentSpec


@cell
def mzi_arm(
    length_y_left: float = 0.8,
    length_y_right: float = 0.8,
    length_x: float = 0.1,
    bend: ComponentSpec = bend_euler,
    straight: ComponentSpec = straight_function,
    straight_x: ComponentSpec | None = None,
    straight_y: ComponentSpec | None = None,
    **kwargs,
) -> Component:
    """Mzi.

    Args:
        length_y_left: vertical length.
        length_y_right: vertical length.
        length_x: horizontal length.
        bend: 90 degrees bend library.
        straight: straight function.
        straight_x: straight for length_x.
        straight_y: straight for length_y.
        kwargs: cross_section settings.

    .. code::

                  B__Lx__B
                  |      |
                  Ly     Lyr
                  |      |
                  B      B
    """
    bend = bend(**kwargs)
    straight_y = straight_y or straight
    straight_x = straight_x or straight
    straight_x = straight_x(length=length_x, **kwargs)

    symbol_to_component = {
        "b": (bend, "o1", "o2"),
        "B": (bend, "o2", "o1"),
        "L": (straight_y(length=length_y_left, **kwargs), "o1", "o2"),
        "R": (
            straight_y(
                length=length_y_right
                + abs(straight_x.get_ports_ysize(port_type="optical")),
                **kwargs,
            ),
            "o1",
            "o2",
        ),
        "-": (straight_x, "o1", "o2"),
    }

    # Each character in the sequence represents a component
    sequence = "bLB-BRb"
    c = component_sequence(sequence=sequence, symbol_to_component=symbol_to_component)

    # Add any electrical ports from references
    for ref_name, ref in c.named_references.items():
        c.add_ports(ref.get_ports_list(port_type="electrical"), prefix=ref_name)

    c.unlock()
    c.auto_rename_ports()
    c.info["length_x"] = length_x
    c.info["length_xsize"] = straight_x.get_ports_xsize()
    return c


if __name__ == "__main__":
    import gdsfactory as gf

    c = mzi_arm(straight_x=gf.components.straight_heater_metal)
    c.pprint_ports()
    c.show(show_ports=True)
