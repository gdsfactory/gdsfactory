from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.components.grating_coupler_elliptical_trenches import grating_coupler_te
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@cell
def grating_coupler_loss_fiber_single(
    grating_coupler: ComponentSpec = grating_coupler_te,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> Component:
    """Returns grating coupler test structure.

    for testing with single fiber input/output

    Args:
        grating_coupler: function.
        cross_section: spec.

    Keyword Args:
        layer_label: for test and measurement label.
        min_input_to_output_spacing: spacing from input to output fiber.
        max_y0_optical: None.
        get_input_labels_function: function to get input labels for grating couplers.
        optical_routing_type: None: autoselection, 0: no extension.
        get_input_label_text_function: for the grating couplers input label.
        get_input_label_text_loopback_function: for the loopbacks input label.
    """
    c = gf.Component()
    grating_coupler = gf.get_component(grating_coupler)

    _ = c << gf.routing.add_fiber_single(
        component=gf.components.straight(cross_section=cross_section),
        grating_coupler=grating_coupler,
        cross_section=cross_section,
        with_loopback=False,
        component_name=grating_coupler.name,
        **kwargs,
    )

    c.copy_child_info(grating_coupler)
    return c


if __name__ == "__main__":
    xs_strip2 = partial(gf.cross_section.strip, layer=(2, 0))
    c = grating_coupler_loss_fiber_single(
        min_input_to_output_spacing=300, cross_section=xs_strip2
    )
    c.show(show_ports=True)
