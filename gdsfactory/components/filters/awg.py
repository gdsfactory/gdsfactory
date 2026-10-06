"""Sample AWG."""

from __future__ import annotations

__all__ = [
    "awg",
    "free_propagation_region",
    "free_propagation_region_input",
    "free_propagation_region_output",
]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name(tags=["filters"])
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

                 length
                 <-->
                   /|
                  / |
           width1|  | width2
                  \ |
                   \|
    """
    return cf.free_propagation_region(
        width1=width1,
        width2=width2,
        length=length,
        wg_width=wg_width,
        inputs=inputs,
        outputs=outputs,
        cross_section=cross_section,
    )


free_propagation_region_input = CellAlias(free_propagation_region, inputs=1)

free_propagation_region_output = CellAlias(
    free_propagation_region, inputs=10, width1=10, width2=20.0
)


@gf.cell_with_module_name(tags=["filters"])
def awg(
    arms: int = 10,
    outputs: int = 3,
    free_propagation_region_input_function: ComponentSpec = "free_propagation_region_input",
    free_propagation_region_output_function: ComponentSpec = "free_propagation_region_output",
    fpr_spacing: float = 50.0,
    arm_spacing: float = 1.0,
    length_increment: float = 0.0,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns an Arrayed Waveguide grating.

    To simulate you can use
    <https://github.com/dnrobin/awg-python>

    Args:
        arms: number of arms.
        outputs: number of outputs.
        free_propagation_region_input_function: for input.
        free_propagation_region_output_function: for output.
        fpr_spacing: x separation between input/output free propagation region.
        arm_spacing: y separation between arms (used when length_increment == 0).
        length_increment: constant length step dL (um); when > 0 the arms form a
            nested fan where arm i is exactly i*dL longer than arm 0 -- the property
            that makes an AWG disperse light by wavelength.
        cross_section: cross_section function.
    """
    return cf.awg(
        arms=arms,
        outputs=outputs,
        free_propagation_region_input_function=free_propagation_region_input_function,
        free_propagation_region_output_function=free_propagation_region_output_function,
        fpr_spacing=fpr_spacing,
        arm_spacing=arm_spacing,
        length_increment=length_increment,
        cross_section=cross_section,
    )


if __name__ == "__main__":
    gf.gpdk.PDK.activate()
    c = awg()
    c.show()
