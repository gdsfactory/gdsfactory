from typing import Tuple

import pp
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.grating_coupler.elliptical import grating_coupler_elliptical_te
from pp.components.straight import straight
from pp.components.straight_array import straight_array
from pp.routing.get_route import get_route_from_waypoints_no_taper
from pp.types import ComponentFactory


@pp.cell
def grating_coupler_tree(
    n_straights: int = 4,
    straight_spacing: int = 4,
    straigth: ComponentFactory = straight,
    grating_coupler_function: ComponentFactory = grating_coupler_elliptical_te,
    with_loop_back: bool = False,
    route_filter: ComponentFactory = get_route_from_waypoints_no_taper,
    bend_factory: ComponentFactory = bend_euler,
    bend_radius: float = 10.0,
    fanout_length: float = 0.0,
    layer_label: Tuple[int, int] = pp.LAYER.LABEL,
    **kwargs
) -> Component:
    """Array of straights connected with grating couplers
    useful to align the 4 corners of the chip

    .. plot::
      :include-source:

      import pp

      c = pp.components.grating_coupler_tree()
      c.plot()

    """
    c = straight_array(
        n_straights=n_straights,
        spacing=straight_spacing,
        straigth=straigth,
    )

    cc = pp.routing.add_fiber_array(
        component=c,
        with_align_ports=with_loop_back,
        optical_routing_type=0,
        grating_coupler=grating_coupler_function,
        bend_radius=bend_radius,
        fanout_length=fanout_length,
        route_filter=route_filter,
        component_name=c.name,
        straight_factory=straigth,
        bend_factory=bend_factory,
        layer_label=layer_label,
        taper_factory=None,
        **kwargs,
    )
    cc.ignore.add("route_filter")
    cc.ignore.add("module")
    return cc


if __name__ == "__main__":
    c = grating_coupler_tree()
    # print(c.get_settings())
    c.show()
