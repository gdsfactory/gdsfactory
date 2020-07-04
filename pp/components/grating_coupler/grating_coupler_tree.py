import pp
from pp.port import deco_rename_ports
from pp.components import waveguide
from pp.components.bend_circular import bend_circular
from pp.components import grating_coupler_elliptical_te
from pp.routing.connect import connect_strip_way_points_no_taper


@pp.autoname
@deco_rename_ports
def _grating_coupler_tree(n_waveguides=4, waveguide_spacing=4, waveguide=waveguide):
    """ array of waveguides connected with grating couplers
    useful to align the 4 corners of the chip
    """

    c = pp.Component()
    w = pp.call_if_func(waveguide)

    for i in range(n_waveguides):
        wref = c.add_ref(w)
        wref.y += i * (waveguide_spacing + w.width)
        c.ports["E" + str(i)] = wref.ports["E0"]
        c.ports["W" + str(i)] = wref.ports["W0"]
    return c


@pp.autoname
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
    """ array of waveguides connected with grating couplers
    useful to align the 4 corners of the chip

    .. plot::
      :include-source:

      import pp

      c = pp.c.grating_coupler_tree()
      pp.plotgds(c)

    """
    c = _grating_coupler_tree(
        n_waveguides=n_waveguides,
        waveguide_spacing=waveguide_spacing,
        waveguide=waveguide,
    )

    cc = pp.routing.add_io_optical(
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
    return cc


if __name__ == "__main__":
    c = grating_coupler_tree()
    print(c.ports)
    pp.show(c)
