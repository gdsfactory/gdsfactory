import pp
from pp.components.bend_circular import bend_circular
from pp.components.grating_coupler.elliptical import grating_coupler_elliptical_te
from pp.components.waveguide import waveguide
from pp.components.waveguide_array import waveguide_array
from pp.routing.connect import connect_strip_way_points_no_taper


@pp.cell
def grating_coupler_tree(
    n_waveguides=4,
    waveguide_spacing=4,
    waveguide=waveguide,
    grating_coupler_function=grating_coupler_elliptical_te,
    with_loop_back=False,
    route_filter=connect_strip_way_points_no_taper,
    bend_factory=bend_circular,
    bend_radius=10.0,
    layer_label=pp.LAYER.LABEL,
    **kwargs
):
    """array of waveguides connected with grating couplers
    useful to align the 4 corners of the chip

    .. plot::
      :include-source:

      import pp

      c = pp.c.grating_coupler_tree()
      pp.plotgds(c)

    """
    c = waveguide_array(
        n_waveguides=n_waveguides, spacing=waveguide_spacing, waveguide=waveguide,
    )

    cc = pp.routing.add_fiber_array(
        c,
        with_align_ports=with_loop_back,
        optical_routing_type=0,
        grating_coupler=grating_coupler_function,
        bend_radius=bend_radius,
        fanout_length=85.0,
        route_filter=route_filter,
        component_name=c.name,
        straight_factory=waveguide,
        bend_factory=bend_factory,
        layer_label=layer_label,
        **kwargs
    )
    cc.ignore.add("route_filter")
    cc.ignore.add("module")
    return cc


if __name__ == "__main__":
    c = grating_coupler_tree()
    print(c.get_settings())
    pp.show(c)
