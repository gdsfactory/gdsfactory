import pp
from pp.add_labels import get_input_label
from pp.components.bend_circular import bend_circular as bend_circular_function
from pp.components.grating_coupler.elliptical_trenches import (
    grating_coupler_te,
    grating_coupler_tm,
)
from pp.components.taper import taper as taper_function
from pp.components.waveguide import waveguide as waveguide_function
from pp.container import container
from pp.routing.manhattan import round_corners
from pp.routing.utils import (
    check_ports_have_equal_spacing,
    direction_ports_from_list_ports,
)


@container
def add_termination(component, terminator=taper_function):
    """ returns component containing a comonent with all ports terminated """
    terminator = pp.call_if_func(terminator)
    c = pp.Component(name=component.name + "_t")
    c.add_ref(component)

    for port in component.ports.values():
        t_ref = c.add_ref(terminator)
        t_ref.connect(list(t_ref.ports.values())[0].name, port)

    return c


def add_gratings_and_loop_back_te(*args, **kwargs):
    return add_gratings_and_loop_back(*args, **kwargs)


def add_gratings_and_loop_back_tm(*args, grating_coupler=grating_coupler_tm, **kwargs):
    return add_gratings_and_loop_back(*args, grating_coupler=grating_coupler, **kwargs)


@container
def add_gratings_and_loop_back(
    component,
    grating_coupler=grating_coupler_te,
    excluded_ports=None,
    grating_separation=127.0,
    bend_radius_align_ports=10.0,
    gc_port_name=None,
    gc_rotation=-90,
    waveguide_separation=5.0,
    bend_factory=bend_circular_function,
    waveguide_factory=waveguide_function,
    layer_label=pp.LAYER.LABEL,
    component_name=None,
    with_loopback=True,
):
    """ returns a component with grating_couplers and loopback
    """
    excluded_ports = excluded_ports or []
    gc = pp.call_if_func(grating_coupler)

    direction = "S"
    component_name = component_name or component.name
    name = f"{component.name}_{gc.polarization}"
    c = pp.Component(name=name)
    c.add_ref(component)

    # Find grating port name if not specified
    if gc_port_name is None:
        gc_port_name = list(gc.ports.values())[0].name

    # List the optical ports to connect
    optical_ports = component.get_ports_list(port_type="optical")
    optical_ports = [p for p in optical_ports if p.name not in excluded_ports]
    optical_ports = direction_ports_from_list_ports(optical_ports)[direction]

    # Check if the ports are equally spaced
    grating_separation_extracted = check_ports_have_equal_spacing(optical_ports)
    if grating_separation_extracted != grating_separation:
        raise ValueError(
            "Grating separation must be {}. Got {}".format(
                grating_separation, grating_separation_extracted
            )
        )

    # Add grating couplers
    couplers = []
    for port in optical_ports:
        coupler_ref = c.add_ref(gc)
        coupler_ref.connect(list(coupler_ref.ports.values())[0].name, port)
        couplers += [coupler_ref]

    # add labels
    for i, optical_port in enumerate(optical_ports):
        label = get_input_label(
            optical_port,
            couplers[i],
            i,
            component_name=component_name,
            layer_label=layer_label,
        )
        c.add(label)

    if with_loopback:
        y0 = couplers[0].ports[gc_port_name].y
        xs = [p.x for p in optical_ports]
        x0 = min(xs) - grating_separation
        x1 = max(xs) + grating_separation

        gca1, gca2 = [
            gc.ref(position=(x, y0), rotation=gc_rotation, port_id=gc_port_name)
            for x in [x0, x1]
        ]

        gsi = gc.size_info
        p0 = gca1.ports[gc_port_name].position
        p1 = gca2.ports[gc_port_name].position
        a = bend_radius_align_ports + 0.5
        b = max(2 * a, grating_separation / 2)
        y_bot_align_route = -gsi.width - waveguide_separation

        route = [
            p0,
            p0 + (0, a),
            p0 + (b, a),
            p0 + (b, y_bot_align_route),
            p1 + (-b, y_bot_align_route),
            p1 + (-b, a),
            p1 + (0, a),
            p1,
        ]
        bend90 = bend_factory(radius=bend_radius_align_ports)
        loop_back = round_corners(route, bend90, waveguide_factory)
        elements = [gca1, gca2, loop_back]
        c.add(elements)
    return c


if __name__ == "__main__":
    # gc = pp.c.grating_coupler_elliptical_te()
    # cc = add_termination(c, gc)
    # import pp
    # c = pp.c.waveguide()
    from pp.components.spiral_inner_io import spiral_inner_io

    c = spiral_inner_io()
    cc = add_gratings_and_loop_back(c, with_loopback=False)

    # cc = add_termination(component=c)
    print(cc.get_settings()["settings"]["component"])
    pp.show(cc)
