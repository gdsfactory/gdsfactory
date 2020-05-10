import pp
from pp.add_labels import get_optical_text
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_te
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_tm
from pp.config import CONFIG


def add_grating_couplers(
    component,
    grating_coupler=grating_coupler_te,
    layer_label=CONFIG["layers"]["LABEL"],
    input_port_indexes=[0],
):
    """ returns component with grating ports and labels on each port """
    component = pp.call_if_func(component)
    c = pp.Component(name=component.name + "_c")
    c.add_ref(component)
    grating_coupler = pp.call_if_func(grating_coupler)

    for i, port in enumerate(component.ports.values()):
        t_ref = c.add_ref(grating_coupler)
        t_ref.connect(list(t_ref.ports.values())[0], port)
        label = get_optical_text(port, grating_coupler, i)
        c.label(label, position=port.midpoint, layer=layer_label)

    return c


def add_te(*args, **kwargs):
    return add_grating_couplers(*args, **kwargs)


def add_tm(*args, grating_coupler=grating_coupler_tm, **kwargs):
    return add_grating_couplers(*args, grating_coupler=grating_coupler, **kwargs)


if __name__ == "__main__":
    # c = pp.c.grating_coupler_elliptical_te()
    # print(c.wavelength)

    # print(c.get_property('wavelength'))

    c = pp.c.waveguide(width=2)
    # cc = add_grating_couplers(c)
    cc = add_tm(c)
    print(cc)
    pp.show(cc)
