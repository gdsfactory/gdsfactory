from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
def mmi(
    inputs: int = 1,
    outputs: int = 4,
    width: float | None = None,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 5,
    gap_input_tapers: float = 0.25,
    gap_output_tapers: float = 0.25,
    taper: ComponentSpec = "taper",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    input_positions: list[float] | None = None,
    output_positions: list[float] | None = None,
) -> Component:
    r"""Mxn MultiMode Interferometer (MMI).

    Args:
        inputs: number of inputs.
        outputs: number of outputs.
        width: input and output straight width. Defaults to cross_section.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_input_tapers: gap between input tapers from edge to edge.
        gap_output_tapers: gap between output tapers from edge to edge.
        taper: taper function.
        straight: straight function.
        cross_section: specification (CrossSection, string or dict).
        input_positions: optional positions of the inputs.
        output_positions: optional positions of the outputs.

    .. code::

                   length_mmi
                    <------>
                    ________
                   |        |
                __/          \__
            o2  __            __  o3
                  \          /_ _ _ _
                  |         | _ _ _ _| gap_output_tapers
                __/          \__
            o1  __            __  o4
                  \          /
                   |________|
                 | |
                 <->
            length_taper
    """
    c = Component()
    gap_input_tapers = gf.snap.snap_to_grid(gap_input_tapers, grid_factor=2)
    gap_output_tapers = gf.snap.snap_to_grid(gap_output_tapers, grid_factor=2)
    w_taper = width_taper
    x = gf.get_cross_section(cross_section)
    xs_mmi = gf.get_cross_section(cross_section, width=width_mmi)
    width = width or x.width

    _taper = gf.get_component(
        taper,
        length=length_taper,
        width1=width,
        width2=w_taper,
        cross_section=cross_section,
    )

    _ = c << gf.get_component(straight, length=length_mmi, cross_section=xs_mmi)
    wg_spacing_input = gap_input_tapers + width_taper
    wg_spacing_output = gap_output_tapers + width_taper

    yi = -(inputs - 1) * wg_spacing_input / 2
    yo = -(outputs - 1) * wg_spacing_output / 2

    input_positions = input_positions or [
        yi + i * wg_spacing_input for i in range(inputs)
    ]
    output_positions = output_positions or [
        yo + i * wg_spacing_output for i in range(outputs)
    ]

    temp_component = Component()

    ports = [
        temp_component.add_port(
            name=f"in_{i}",
            orientation=180,
            center=(0, y),
            width=w_taper,
            layer=gf.get_layer(x.layer),
            cross_section=x,
        )
        for i, y in enumerate(input_positions)
    ]

    ports += [
        temp_component.add_port(
            name=f"out_{i}",
            orientation=0,
            center=(+length_mmi, y),
            width=w_taper,
            layer=gf.get_layer(x.layer),
            cross_section=x,
        )
        for i, y in enumerate(output_positions)
    ]

    for port in ports:
        taper_ref = c << _taper
        taper_ref.connect("o2", port, allow_width_mismatch=True)
        c.add_port(name=port.name, port=taper_ref.ports["o1"])

    x.add_bbox(c)
    c.auto_rename_ports()
    c.flatten()
    return c


if __name__ == "__main__":
    c = mmi(inputs=1, outputs=4)
    c.show()
