import pp
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_te
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_tm
from pp.routing.get_input_labels import get_input_labels
from pp.container import container


@container
def add_grating_couplers(
    component,
    grating_coupler=grating_coupler_te,
    layer_label=pp.LAYER.LABEL,
    gc_port_name="W0",
    get_input_labels_function=get_input_labels,
):
    """ returns component with grating ports and labels on each port
    """
    component = pp.call_if_func(component)
    c = pp.Component(name=component.name + "_c")
    c.add_ref(component)
    grating_coupler = pp.call_if_func(grating_coupler)

    io_gratings = []
    for i, port in enumerate(component.ports.values()):
        gc_ref = grating_coupler.ref()
        gc_ref.connect(list(gc_ref.ports.values())[0], port)
        io_gratings.append(gc_ref)
        c.add(gc_ref)

    labels = get_input_labels_function(
        io_gratings,
        list(component.ports.values()),
        component_name=component.name,
        layer_label=layer_label,
        gc_port_name=gc_port_name,
    )
    c.add(labels)

    return c


def add_te(*args, **kwargs):
    return add_grating_couplers(*args, **kwargs)


def add_tm(*args, grating_coupler=grating_coupler_tm, **kwargs):
    return add_grating_couplers(*args, grating_coupler=grating_coupler, **kwargs)


if __name__ == "__main__":
    # from pp.add_labels import get_optical_text
    # c = pp.c.grating_coupler_elliptical_te()
    # print(c.wavelength)

    # print(c.get_property('wavelength'))

    c = pp.c.waveguide(width=2)
    # cc = add_grating_couplers(c)
    cc = add_tm(c)
    print(cc)
    pp.show(cc)
