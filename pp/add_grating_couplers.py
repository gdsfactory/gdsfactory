"""Add grating_couplers to a component."""
import pp
from pp.component import Component
from pp.components.grating_coupler.elliptical_trenches import (
    grating_coupler_te,
    grating_coupler_tm,
)
from pp.container import container
from pp.routing.get_input_labels import get_input_labels


@container
def add_grating_couplers(
    component: Component,
    grating_coupler=grating_coupler_te,
    layer_label=pp.LAYER.LABEL,
    gc_port_name: str = "W0",
    get_input_labels_function=get_input_labels,
):
    """Return component with grating couplers and labels."""

    cnew = Component(name=component.name + "_c")
    cnew.add_ref(component)
    grating_coupler = pp.call_if_func(grating_coupler)

    io_gratings = []
    for port in component.ports.values():
        gc_ref = grating_coupler.ref()
        gc_ref.connect(list(gc_ref.ports.values())[0], port)
        io_gratings.append(gc_ref)
        cnew.add(gc_ref)

    labels = get_input_labels_function(
        io_gratings,
        list(component.ports.values()),
        component_name=component.name,
        layer_label=layer_label,
        gc_port_name=gc_port_name,
    )
    cnew.add(labels)
    return cnew


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
