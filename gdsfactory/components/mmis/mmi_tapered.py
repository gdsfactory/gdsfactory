from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.tapers.taper import taper as taper_function
from gdsfactory.typings import ComponentFactory, CrossSectionSpec


@gf.cell
def mmi_tapered(
    inputs: int = 1,
    outputs: int = 2,
    width: float | None = None,
    width_taper_in: float = 2.0,
    length_taper_in: float = 1.0,
    width_taper_out: float | None = None,
    length_taper_out: float | None = None,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_taper_start: float | None = None,
    length_taper_end: float | None = None,
    length_mmi: float = 5.5,
    width_mmi: float = 5,
    width_mmi_inner: float | None = None,
    gap_input_tapers: float = 0.25,
    gap_output_tapers: float = 0.25,
    taper: ComponentFactory = taper_function,
    cross_section: CrossSectionSpec = "strip",
    input_positions: list[float] | None = None,
    output_positions: list[float] | None = None,
) -> Component:
    r"""Mxn MultiMode Interferometer (MMI).

    This is jut a more general version of the mmi component.
    Make sure you simulate and optimize the component before using it.

    Args:
        inputs: number of inputs.
        outputs: number of outputs.
        width: input and output straight width. Defaults to cross_section.
        width_taper_in: interface between input straights and mmi region.
        length_taper_in: into the mmi region.
        width_taper_out: interface between mmi region and output straights.
        length_taper_out: into the mmi region.
        width_taper: interface between mmi region and output straights.
        length_taper: into the mmi region.
        length_taper_start: length of the taper at the start. Defaults to length_taper.
        length_taper_end: length of the taper at the end. Defaults to length_taper.
        length_mmi: in x direction.
        width_mmi: in y direction.
        width_mmi_inner: allows adding a different width for the inner mmi region.
        gap_input_tapers: gap between input tapers from edge to edge.
        gap_output_tapers: gap between output tapers from edge to edge.
        taper: taper function.
        cross_section: specification (CrossSection, string or dict).
        input_positions: optional positions of the inputs.
        output_positions: optional positions of the outputs.

    .. code::

                                       ┌───────────┐
                                       │           ├───────────────┐
                                       │           │               ├────────────┐
               width_taper             │           │               │            │
                    ▲ ┌────────────────┤           │               ├────────────┘
                    │ │                │           ├───────────────┘
        ┌───────────┼─┤                │           │
        │           │ │                │           │
        ◄───────────┼─►                │           ├───────────────┐
        └───────────┼─┐                │           │               ├─────────────┐
                    ▼ └────────────────┤           │               │             │
                      ◄───────────────►│           │               ├─────────────┘
        length_taper    length_taper_in│           ├───────────────┘ length_taper
        ◄────────────►                 └───────────┘◄────────────►  ◄────────────►
            start                                  length_taper_out      end
                                       ◄───────────►
                                        length_mmi
    """
    c = Component()
    gap_input_tapers = gf.snap.snap_to_grid(gap_input_tapers, grid_factor=2)
    gap_output_tapers = gf.snap.snap_to_grid(gap_output_tapers, grid_factor=2)
    x = gf.get_cross_section(cross_section)
    width = width or x.width
    width_taper_out = width_taper_out or width_taper_in

    _taper_in = taper(
        length=length_taper_in,
        width1=width_taper,
        width2=width_taper_in,
        cross_section=cross_section,
    )
    _taper_out = taper(
        length=length_taper_out or length_taper_in,
        width2=width_taper_out,
        width1=width_taper,
        cross_section=cross_section,
    )
    _taper_start = taper(
        length=length_taper_start or length_taper,
        width1=width,
        width2=width_taper,
        cross_section=cross_section,
    )

    _taper_end = taper(
        length=length_taper_end or length_taper,
        width2=width_taper,
        width1=width,
        cross_section=cross_section,
    )

    width_mmi_inner = width_mmi_inner or width_mmi

    # _ = c << straight(length=length_mmi, cross_section=xs_mmi)
    mmi_left = c << taper(
        length=length_mmi / 2,
        width1=width_mmi,
        width2=width_mmi_inner,
        cross_section=cross_section,
    )
    mmi_right = c << taper(
        length=length_mmi / 2,
        width1=width_mmi_inner,
        width2=width_mmi,
        cross_section=cross_section,
    )
    mmi_right.connect("o1", mmi_left.ports["o2"])

    wg_spacing_input = gap_input_tapers + width_taper_in
    wg_spacing_output = gap_output_tapers + width_taper_out

    yi = -(inputs - 1) * wg_spacing_input / 2
    yo = -(outputs - 1) * wg_spacing_output / 2

    input_positions = input_positions or [
        yi + i * wg_spacing_input for i in range(inputs)
    ]
    output_positions = output_positions or [
        yo + i * wg_spacing_output for i in range(outputs)
    ]

    temp_component = Component()

    in_ports = [
        temp_component.add_port(
            name=f"in_{i}",
            orientation=180,
            center=(0, y),
            width=width_taper_in,
            layer=gf.get_layer(x.layer),
            cross_section=x,
        )
        for i, y in enumerate(input_positions)
    ]

    out_ports = [
        temp_component.add_port(
            name=f"out_{i}",
            orientation=0,
            center=(+length_mmi, y),
            width=width_taper_out,
            layer=gf.get_layer(x.layer),
            cross_section=x,
        )
        for i, y in enumerate(output_positions)
    ]

    for port in in_ports:
        taper_ref = c << _taper_in
        taper_ref.connect("o2", port, allow_width_mismatch=True)
        taper_outer_ref = c << _taper_start
        taper_outer_ref.connect("o2", taper_ref["o1"], allow_width_mismatch=True)
        c.add_port(name=port.name, port=taper_outer_ref.ports["o1"])

    for port in out_ports:
        taper_ref = c << _taper_out
        taper_ref.connect("o2", port, allow_width_mismatch=True)
        taper_outer_ref = c << _taper_end
        taper_outer_ref.connect("o2", taper_ref["o1"], allow_width_mismatch=True)
        c.add_port(name=port.name, port=taper_outer_ref.ports["o1"])

    x.add_bbox(c)
    c.auto_rename_ports()
    c.flatten()
    return c


if __name__ == "__main__":
    # c = gf.Component()
    # s = c << gf.c.straight()
    # b = c << gf.c.bend_circular()
    # # b.dmirror()
    # b.connect("o1", s.ports["o1"])
    c = mmi_tapered(width_taper_in=5, width_taper_out=2, cross_section="rib")
    c.show()
