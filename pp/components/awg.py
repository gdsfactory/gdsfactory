"""Sample AWG."""

from typing import Iterable, Optional

import numpy as np

import pp
from pp.cell import cell
from pp.component import Component
from pp.tech import TECH_SILICON_C, Tech
from pp.types import Layer


@cell
def free_propagation_region(
    width1: float = 2.0,
    width2: float = 20.0,
    length: float = 20.0,
    wg_width: float = 0.5,
    inputs: int = 1,
    outputs: int = 10,
    layer: Layer = TECH_SILICON_C.layer_wg,
    layers_cladding: Optional[Iterable[Layer]] = None,
    cladding_offset: Optional[float] = None,
    tech: Optional[Tech] = None,
    wg_margin: float = 1.0,
) -> Component:
    r"""

    .. code::

                 length
                 <-->
                   /|
                  / |
           width1|  | width2
                  \ |
                   \|
    """
    tech = tech or TECH_SILICON_C
    y1 = width1 / 2
    y2 = width2 / 2
    o = cladding_offset or tech.cladding_offset
    layers_cladding = layers_cladding or []

    xpts = [0, length, length, 0]
    ypts = [y1, y2, -y2, -y1]

    c = pp.Component()
    c.add_polygon((xpts, ypts), layer=layer)

    if inputs == 1:
        c.add_port(
            "W0",
            midpoint=(0, 0),
            width=wg_width,
            orientation=180,
            layer=layer,
        )
    else:
        y = np.linspace(-width1 / 2 + wg_width / 2, width1 / 2 - wg_width / 2, inputs)
        y = pp.snap_to_grid(y)
        for i, y in enumerate(y):
            c.add_port(
                f"W{i}",
                midpoint=(0, y),
                width=wg_width,
                orientation=0,
                layer=layer,
            )

    y = np.linspace(-width2 / 2 + wg_width / 2, width2 / 2 - wg_width / 2, outputs)
    y = pp.snap_to_grid(y)
    for i, y in enumerate(y):
        c.add_port(
            f"E{i}",
            midpoint=(length, y),
            width=wg_width,
            orientation=0,
            layer=layer,
        )

    ypts = [y1 + o, y2 + o, -y2 - o, -y1 - o]

    for layer in layers_cladding:
        c.add_polygon((xpts, ypts), layer=layer)

    c.info["length"] = length
    c.info["width1"] = width1
    c.info["width2"] = width2
    return c


@cell
def free_propagation_region_input(inputs: int = 1, **kwargs) -> Component:
    return free_propagation_region(inputs=inputs, **kwargs)


@cell
def free_propagation_region_output(
    inputs: int = 10, width1: float = 10.0, width2: float = 20.0, **kwargs
) -> Component:
    return free_propagation_region(
        inputs=inputs, width2=width2, width1=width1, **kwargs
    )


@cell
def awg(
    arms: int = 10,
    outputs: int = 3,
    free_propagation_region_input_function=free_propagation_region_input,
    free_propagation_region_output_function=free_propagation_region_output,
    fpr_spacing: float = 50.0,
) -> Component:
    """Returns a basic Arrayed Waveguide grating.

    Args:
        arms: number of arms
        outputs: number of outputs
        free_propagation_region_input_function: for input
        free_propagation_region_output_function: for output
        fpr_spacing: x separation between input/output FPR

    """
    c = Component()
    fpr_in = free_propagation_region_input_function(
        inputs=1,
        outputs=arms,
    )
    fpr_out = free_propagation_region_output_function(
        inputs=outputs,
        outputs=arms,
    )

    fpr_in_ref = c.add_ref(fpr_in)
    fpr_out_ref = c.add_ref(fpr_out)

    fpr_in_ref.rotate(90)
    fpr_out_ref.rotate(90)

    fpr_out_ref.x += fpr_spacing
    routes = pp.routing.get_bundle(
        fpr_in_ref.get_ports_list(prefix="E"), fpr_out_ref.get_ports_list(prefix="E")
    )

    c.lengths = []
    for route in routes:
        c.add(route["references"])
        c.lengths.append(route["length"])

    c.add_port("W0", port=fpr_in_ref.ports["W0"])

    for i, port in enumerate(fpr_out_ref.get_ports_list(prefix="W")):
        c.add_port(f"E{i}", port=port)

    c.delta_length = np.mean(np.diff(c.lengths))
    return c


if __name__ == "__main__":
    c = free_propagation_region(inputs=2, outputs=4)
    # print(c.ports.keys())
    c = awg()
    c.show()
