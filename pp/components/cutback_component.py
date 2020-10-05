import pp
from pp.components import bend_euler180
from pp.components.component_sequence import component_sequence


@pp.autoname
def cutback_component(
    component,
    cols=4,
    rows=5,
    bend_radius=10,
    port1_id="W0",
    port2_id="E0",
    middle_couples=2,
):
    """ Flips the component, good for tapers that end in wide waveguides
    Args:
        component
        cols
        rows

    """
    bend180 = bend_euler180(radius=bend_radius)

    # Define a map between symbols and (component, input port, output port)
    string_to_device_in_out_ports = {
        "A": (component, port1_id, port2_id),
        "B": (component, port2_id, port1_id),
        "D": (bend180, "W0", "W1"),
        "C": (bend180, "W1", "W0"),
    }

    # Generate the sequence of staircases

    s = ""
    for i in range(rows):
        s += "AB" * cols
        s += "D" if i % 2 == 0 else "C"

    s = s[:-1]
    s += "AB" * middle_couples

    for i in range(rows):
        s += "AB" * cols
        s += "D" if (i + rows) % 2 == 0 else "C"

    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(s, string_to_device_in_out_ports)
    c.update_settings(n_devices=len(s))
    return c


@pp.autoname
def cutback_component_flipped(
    component,
    cols=4,
    rows=5,
    bend_radius=10,
    port1_id="E0",
    port2_id="W0",
    middle_couples=2,
):
    bend180 = bend_euler180(radius=bend_radius)

    # Define a map between symbols and (component, input port, output port)
    string_to_device_in_out_ports = {
        "A": (component, port1_id, port2_id),
        "B": (component, port2_id, port1_id),
        "D": (bend180, "W0", "W1"),
        "C": (bend180, "W1", "W0"),
    }

    # Generate the sequence of staircases

    s = ""
    for i in range(rows):
        s += "AB" * cols
        s += "C" if i % 2 == 0 else "D"

    s = s[:-1]
    s += "AB" * middle_couples

    for i in range(rows):
        s += "AB" * cols
        s += "D" if (i + rows + 1) % 2 == 0 else "C"

    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(s, string_to_device_in_out_ports)
    c.update_settings(n_devices=len(s))
    return c


@pp.autoname
def cutback_polarization_rotator(n_devices_target, design=3):
    """ sample of component cutback """
    rows = 4
    cols = n_devices_target // (rows * 2)
    from pp.components.polarization_rotator import polarization_rotator

    c = cutback_component(
        component=polarization_rotator(design=design), rows=rows, cols=cols
    )
    cc = pp.routing.add_fiber_array(
        c,
        grating_coupler=pp.c.grating_coupler_elliptical_tm,
        optical_routing_type=0,
        connected_port_list_ids=["out", "in"],
    )

    return cc


if __name__ == "__main__":
    pass
