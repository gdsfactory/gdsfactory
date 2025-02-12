"""Sample AWG."""

from __future__ import annotations

from functools import partial

import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def free_propagation_region(
    width1: float = 2.0,
    width2: float = 20.0,
    length: float = 20.0,
    wg_width: float = 0.5,
    inputs: int = 1,
    outputs: int = 10,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""Free propagation region.

    Args:
        width1: width of the input region.
        width2: width of the output region.
        length: length of the free propagation region.
        wg_width: waveguide width.
        inputs: number of inputs.
        outputs: number of outputs.
        cross_section: cross_section function.

    .. code::

                 length
                 <-->
                   /|
                  / |
           width1|  | width2
                  \ |
                   \|
    """
    y1 = width1 / 2
    y2 = width2 / 2
    xs = gf.get_cross_section(cross_section)
    o = 0
    layer = xs.layer
    assert layer is not None

    xpts = [0, length, length, 0]
    ypts = [y1, y2, -y2, -y1]

    c = gf.Component()
    c.add_polygon(list(zip(xpts, ypts)), layer=layer)

    if inputs == 1:
        c.add_port(
            "o1",
            center=(0, 0),
            width=wg_width,
            orientation=180,
            layer=layer,
        )
    else:
        y = np.linspace(-width1 / 2 + wg_width / 2, width1 / 2 - wg_width / 2, inputs)
        y = gf.snap.snap_to_grid(y)
        for i, yi in enumerate(y):
            c.add_port(
                f"W{i}",
                center=(0, float(yi)),
                width=wg_width,
                orientation=180,
                layer=layer,
            )

    y = np.linspace(-width2 / 2 + wg_width / 2, width2 / 2 - wg_width / 2, outputs)
    y = gf.snap.snap_to_grid(y)
    for i, yi in enumerate(y):
        c.add_port(
            f"E{i}",
            center=(length, float(yi)),
            width=wg_width,
            orientation=0,
            layer=layer,
        )

    ypts = [y1 + o, y2 + o, -y2 - o, -y1 - o]

    c.info["length"] = length
    c.info["width1"] = width1
    c.info["width2"] = width2
    return c


free_propagation_region_input = partial(free_propagation_region, inputs=1)

free_propagation_region_output = partial(
    free_propagation_region, inputs=10, width1=10, width2=20.0
)


@gf.cell
def awg(
    arms: int = 10,
    outputs: int = 3,
    free_propagation_region_input_function: ComponentSpec = free_propagation_region_input,
    free_propagation_region_output_function: ComponentSpec = free_propagation_region_output,
    fpr_spacing: float = 50.0,
    arm_spacing: float = 1.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns a basic Arrayed Waveguide grating.

    To simulate you can use
    https://github.com/dnrobin/awg-python

    Args:
        arms: number of arms.
        outputs: number of outputs.
        free_propagation_region_input_function: for input.
        free_propagation_region_output_function: for output.
        fpr_spacing: x separation between input/output free propagation region.
        arm_spacing: y separation between arms.
        cross_section: cross_section function.
    """
    c = Component()
    fpr_in = gf.get_component(
        free_propagation_region_input_function,
        inputs=1,
        outputs=arms,
        cross_section=cross_section,
    )
    fpr_out = gf.get_component(
        free_propagation_region_output_function,
        inputs=outputs,
        outputs=arms,
        cross_section=cross_section,
    )

    fpr_in_ref = c.add_ref(fpr_in)
    fpr_out_ref = c.add_ref(fpr_out)

    fpr_in_ref.drotate(90)
    fpr_out_ref.drotate(90)

    fpr_out_ref.dx += fpr_spacing
    _ = gf.routing.route_bundle(
        c,
        gf.port.get_ports_list(fpr_out_ref, prefix="E"),
        gf.port.get_ports_list(fpr_in_ref, prefix="E"),
        sort_ports=True,
        separation=arm_spacing,
        cross_section=cross_section,
    )

    c.add_port("o1", port=fpr_in_ref.ports["o1"])

    for i, port in enumerate(gf.port.get_ports_list(fpr_out_ref, prefix="W")):
        c.add_port(f"E{i}", port=port)

    return c


if __name__ == "__main__":
    # c = free_propagation_region(inputs=2, outputs=4)
    c = awg()
    c.show()
