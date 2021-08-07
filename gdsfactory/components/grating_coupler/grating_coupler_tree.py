from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.grating_coupler.elliptical import (
    grating_coupler_elliptical_te,
)
from gdsfactory.components.straight_array import straight_array
from gdsfactory.tech import TECH
from gdsfactory.types import ComponentFactory


@gf.cell
def grating_coupler_tree(
    n: int = 4,
    straight_spacing: int = 4,
    grating_coupler_function: ComponentFactory = grating_coupler_elliptical_te,
    with_loopback: bool = False,
    bend_factory: ComponentFactory = bend_euler,
    fanout_length: float = 0.0,
    layer_label: Tuple[int, int] = TECH.layer_label,
    waveguide: str = "strip",
    **kwargs
) -> Component:
    """Array of straights connected with grating couplers
    useful to align the 4 corners of the chip

    Args:
        waveguide
        kwargs: waveguide_settings

    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.grating_coupler_tree()
      c.plot()

    """
    c = straight_array(
        n=n,
        spacing=straight_spacing,
        waveguide=waveguide,
        **kwargs,
    )

    cc = gf.routing.add_fiber_array(
        component=c,
        with_loopback=with_loopback,
        optical_routing_type=0,
        grating_coupler=grating_coupler_function,
        fanout_length=fanout_length,
        component_name=c.name,
        bend_factory=bend_factory,
        layer_label=layer_label,
        taper_factory=None,
        waveguide=waveguide,
        **kwargs,
    )
    cc.ignore.add("route_filter")
    cc.ignore.add("module")
    return cc


if __name__ == "__main__":
    c = grating_coupler_tree(waveguide="nitride")
    # print(c.get_settings())
    c.show()
