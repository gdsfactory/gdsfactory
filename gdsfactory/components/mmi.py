from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.component import Component
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper as taper_function
from gdsfactory.typings import ComponentFactory, CrossSectionSpec


@gf.cell
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
    taper: ComponentFactory = taper_function,
    straight: ComponentFactory = straight_function,
    with_bbox: bool = True,
    cross_section: CrossSectionSpec = "strip",
    input_positions: list[float] | None = None,
    output_positions: list[float] | None = None,
) -> Component:
    r"""mxn MultiMode Interferometer (MMI).

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
        with_bbox: add rectangular box in cross_section
            bbox_layers and bbox_offsets to avoid DRC sharp edges.
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
    w_mmi = width_mmi
    w_taper = width_taper
    x = gf.get_cross_section(cross_section)
    width = width or x.width

    _taper = taper(
        length=length_taper,
        width1=width,
        width2=w_taper,
        cross_section=cross_section,
        add_pins=None,
        add_bbox=None,
        decorator=None,
    )

    mmi = c << straight(
        length=length_mmi,
        width=w_mmi,
        cross_section=cross_section,
        add_pins=None,
        add_bbox=None,
        decorator=None,
    )

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

    ports = [
        gf.Port(
            f"in_{i}",
            orientation=180,
            center=(0, y),
            width=w_taper,
            layer=x.layer,
            cross_section=x,
        )
        for i, y in enumerate(input_positions)
    ]

    ports += [
        gf.Port(
            f"out_{i}",
            orientation=0,
            center=(+length_mmi, y),
            width=w_taper,
            layer=x.layer,
            cross_section=x,
        )
        for i, y in enumerate(output_positions)
    ]

    for port in ports:
        taper_ref = c << _taper
        taper_ref.connect(port="o2", destination=port)
        c.add_port(name=port.name, port=taper_ref.ports["o1"])
        c.absorb(taper_ref)

    if with_bbox:
        padding = []
        for offset in x.bbox_offsets:
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                top=offset,
            )
            padding.append(points)

        for layer, points in zip(x.bbox_layers, padding):
            c.add_polygon(points, layer=layer)

    c.absorb(mmi)
    if x.add_bbox:
        c = x.add_bbox(c)
    if x.add_pins:
        c = x.add_pins(c)
    if x.decorator:
        c = x.decorator(c)
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    # import gdsfactory as gf
    # c = gf.components.mmi1x2(cross_section="rib_conformal")
    c = mmi(inputs=2, outputs=4, gap_input_tapers=0.5, input_positions=[-1, 1])
    print(len(c.ports))
    c.show(show_ports=True)
