from typing import Tuple

from pydantic import validate_arguments

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.grating_coupler.elliptical import grating_coupler_elliptical_te
from pp.components.straight_array import straight_array
from pp.tech import TECH
from pp.types import ComponentFactory


@cell
@validate_arguments
def grating_coupler_tree(
    n_straights: int = 4,
    straight_spacing: int = 4,
    grating_coupler_function: ComponentFactory = grating_coupler_elliptical_te,
    with_loop_back: bool = False,
    bend_factory: ComponentFactory = bend_euler,
    fanout_length: float = 0.0,
    layer_label: Tuple[int, int] = TECH.layer_label,
    cross_section_name: str = "strip",
    **kwargs
) -> Component:
    """Array of straights connected with grating couplers
    useful to align the 4 corners of the chip

    Args:
        cross_section_name
        kwargs: cross_section_settings

    .. plot::
      :include-source:

      import pp

      c = pp.components.grating_coupler_tree()
      c.plot()

    """
    c = straight_array(
        n_straights=n_straights,
        spacing=straight_spacing,
        cross_section_name=cross_section_name,
        **kwargs,
    )

    cc = pp.routing.add_fiber_array(
        component=c,
        with_align_ports=with_loop_back,
        optical_routing_type=0,
        grating_coupler=grating_coupler_function,
        fanout_length=fanout_length,
        component_name=c.name,
        bend_factory=bend_factory,
        layer_label=layer_label,
        taper_factory=None,
        cross_section_name=cross_section_name,
        **kwargs,
    )
    cc.ignore.add("route_filter")
    cc.ignore.add("module")
    return cc


if __name__ == "__main__":
    c = grating_coupler_tree(cross_section_name="nitride")
    # print(c.get_settings())
    c.show()
