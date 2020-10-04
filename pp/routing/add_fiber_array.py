from pp.routing.connect_component import add_io_optical


def add_fiber_array(*args, **kwargs):
    """returns component with optical IO (tapers, south routes and grating_couplers)

    Args:
        component: to connect
        optical_io_spacing: SPACING_GC
        grating_coupler: grating coupler instance, function or list of functions
        bend_factory: bend_circular
        straight_factory: waveguide
        fanout_length: None  # if None, automatic calculation of fanout length
        max_y0_optical: None
        with_align_ports: True, adds loopback structures
        waveguide_separation: 4.0
        bend_radius: BEND_RADIUS
        list_port_labels: None, adds TM labels to port indices in this list
        connected_port_list_ids: None # only for type 0 optical routing
        nb_optical_ports_lines: 1
        force_manhattan: False
        excluded_ports:
        grating_indices: None
        routing_waveguide: None
        routing_method: connect_strip
        gc_port_name: W0
        optical_routing_type: None: autoselection, 0: no extension
        gc_rotation: -90
        layer_label: LAYER.LABEL
        input_port_indexes: [0]
        component_name: for the label
        taper_factory: taper

    .. plot::
      :include-source:

       import pp
       from pp.routing import add_io_optical

       c = pp.c.mmi1x2()
       cc = add_io_optical(c)
       pp.plotgds(cc)

    """

    return add_io_optical(*args, **kwargs)


if __name__ == "__main__":
    import pp

    c = pp.c.crossing()
    cc = add_fiber_array(c)
    pp.show(cc)
