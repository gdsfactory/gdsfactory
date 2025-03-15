from __future__ import annotations

from typing import cast

import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.snap import snap_to_grid2x
from gdsfactory.typings import CrossSectionSpec, Size


def _generate_fins(
    c: Component,
    fin_size: Size,
    taper_length: float,
    length: float,
    cross_section: CrossSectionSpec,
) -> Component:
    """Generates fins on the input/output straights.

    Args:
        c: Component.
        fin_size: Specifies the x- and y-size of the `fins`.
        taper_length: between the input/output straight and the DBR region.
        length: Length of the DBR region.
        cross_section: CrossSectionSpec.
    """
    xs = gf.get_cross_section(cross_section=cross_section)
    num_fins = xs.width // (2 * fin_size[1])
    x0, y0 = (
        0,
        -num_fins * (2 * fin_size[1]) / 2.0 + fin_size[1] / 2.0,
    )
    xend = 2 * taper_length + length
    assert xs.layer is not None
    for i in range(int(num_fins)):
        y = y0 + i * 2 * fin_size[1]
        rectangle = gf.components.rectangle(
            size=(fin_size[0], fin_size[1]),
            layer=xs.layer,
            centered=True,
            port_type=None,
            port_orientations=None,
        )
        rectangle_input = c << rectangle
        rectangle_input.dmove(
            origin=(x0, y0),
            destination=(
                x0 + fin_size[0] / 2.0 - (2 * taper_length) / 2.0,
                y0 + y + fin_size[1] / 2.0,
            ),
        )

        rectangle_output = c << rectangle
        rectangle_output.dmove(
            origin=(x0, y0),
            destination=(
                xend - fin_size[0] / 2.0 - (2 * taper_length) / 2.0,
                y0 + y + fin_size[1] / 2.0,
            ),
        )
    return c


@gf.cell
def dbr_tapered(
    length: float = 10.0,
    period: float = 0.85,
    dc: float = 0.5,
    w1: float = 0.4,
    w2: float = 1.0,
    taper_length: float = 20.0,
    fins: bool = False,
    fin_size: Size = (0.2, 0.05),
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Distributed Bragg Reflector Cell class.

    Tapers the input straight to a
    periodic straight structure with varying width (1-D photonic crystal).

    Args:
       length: Length of the DBR region.
       period: Period of the repeated unit.
       dc: Duty cycle of the repeated unit (must be a float between 0 and 1.0).
       w1: thin section width. w1 = 0 corresponds to disconnected periodic blocks.
       w2: wide section width.
       taper_length: between the input/output straight and the DBR region.
       fins: If `True`, adds fins to the input/output straights.
       fin_size: Specifies the x- and y-size of the `fins`. Defaults to 200 nm x 50 nm
       cross_section: cross_section spec.

    .. code::

                 period
        <-----><-------->
                _________
        _______|

          w1       w2       ...  n times
        _______
               |_________
    """
    c = gf.Component()

    xs = gf.get_cross_section(cross_section=cross_section, width=w2)

    input_taper = c << gf.components.taper(
        length=taper_length,
        width1=xs.width,
        width2=w1,
        cross_section=cross_section,
    )

    straight = c << gf.components.straight(
        length=length, cross_section=cross_section, width=w1
    )
    straight.dx = 0
    straight.dy = 0

    output_taper = c << gf.components.taper(
        length=taper_length,
        width1=w1,
        width2=xs.width,
        cross_section=cross_section,
    )

    input_taper.connect("o2", straight.ports["o1"])
    output_taper.connect("o1", straight.ports["o2"])
    num = (2 * taper_length + length) // period

    size = cast(tuple[float, float], tuple(snap_to_grid2x((period * dc, w2))))
    assert xs.layer is not None
    teeth = gf.components.rectangle(size=size, layer=xs.layer, port_type=None)

    periodic_structures = c << gf.components.array(
        component=teeth, columns=int(num), column_pitch=period
    )
    periodic_structures.dx = 0
    periodic_structures.dy = 0

    if fins:
        _generate_fins(
            c=c,
            fin_size=fin_size,
            taper_length=taper_length,
            length=length,
            cross_section=xs,
        )

    xs.add_bbox(c)
    c.add_port("o1", port=input_taper.ports["o1"])
    c.add_port("o2", port=output_taper.ports["o2"])
    return c


if __name__ == "__main__":
    # c = dbr_tapered(length=10, period=0.85, dc=0.5, w2=1, w1=0.4, taper_length=20, fins=True)
    c = dbr_tapered(cross_section="rib")
    c.show()
