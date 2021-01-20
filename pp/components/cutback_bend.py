import pp
from pp.components.bend_circular import bend_circular
from pp.components.component_sequence import component_sequence
from pp.components.euler.bend_euler import bend_euler90, bend_euler180
from pp.components.waveguide import waveguide
from pp.routing.add_fiber_array import add_fiber_array


def _get_bend_size(bend90):
    p1, p2 = list(bend90.ports.values())[:2]
    bsx = abs(p2.x - p1.x)
    bsy = abs(p2.y - p1.y)
    return max(bsx, bsy)


@pp.cell
def cutback_bend(bend90, straight_length=5.0, n_steps=6, n_stairs=5):
    """ Deprecated! use cutback_bend90 instead!
    this is a stair

    .. code::
            _
          _|
        _|

        _ this is a step

    """

    wg = waveguide(length=straight_length, width=bend90.ports["W0"].width)

    # Define a map between symbols and (component, input port, output port)
    string_to_device_in_out_ports = {
        "A": (bend90, "W0", "N0"),
        "B": (bend90, "N0", "W0"),
        "S": (wg, "W0", "E0"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(n_stairs):
        s += "ASBS" * n_steps
        s += "ASAS" if i % 2 == 0 else "BSBS"
    s = s[:-4]

    # Create the component from the sequence
    c = component_sequence(s, string_to_device_in_out_ports, start_orientation=90)
    c.update_settings(n_bends=n_steps * n_stairs * 2 + n_stairs * 2 - 2)
    return c


@pp.cell
def cutback_bend90(
    bend90=bend_euler90,
    straight_length=5.0,
    n_steps=6,
    cols=6,
    spacing=5,
    wg_loop_length=None,
    waveguide_factory=waveguide,
):
    """

    .. code::

           _
        |_| |

    """
    bend90 = pp.call_if_func(bend90)

    wg = waveguide_factory(length=straight_length, width=bend90.ports["W0"].width)
    if wg_loop_length is None:
        wg_loop_length = 2 * _get_bend_size(bend90) + spacing + straight_length

    wg_loop = waveguide_factory(length=wg_loop_length, width=bend90.ports["W0"].width,)
    # Define a map between symbols and (component, input port, output port)
    string_to_device_in_out_ports = {
        "A": (bend90, "W0", "N0"),
        "B": (bend90, "N0", "W0"),
        "-": (wg, "W0", "E0"),
        "|": (wg_loop, "W0", "E0"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(cols):
        if i % 2 == 0:  # even row
            s += "A-A-B-B-" * n_steps + "|"
        else:
            s += "B-B-A-A-" * n_steps + "|"
    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(s, string_to_device_in_out_ports, start_orientation=0)
    c.update_settings(n_bends=n_steps * cols * 4)
    return c


def staircase(
    bend90=bend_euler90,
    length_v=5.0,
    length_h=5.0,
    n_steps=4,
    waveguide_factory=waveguide,
):
    bend90 = pp.call_if_func(bend90)

    wgh = waveguide_factory(length=length_h, width=bend90.ports["W0"].width)
    wgv = waveguide_factory(length=length_v, width=bend90.ports["W0"].width)

    # Define a map between symbols and (component, input port, output port)
    string_to_device_in_out_ports = {
        "A": (bend90, "W0", "N0"),
        "B": (bend90, "N0", "W0"),
        "-": (wgh, "W0", "E0"),
        "|": (wgv, "W0", "E0"),
    }

    # Generate the sequence of staircases
    s = "-A|B" * n_steps + "-"

    # Create the component from the sequence
    c = component_sequence(s, string_to_device_in_out_ports, start_orientation=0)
    c.update_settings(n_bends=2 * n_steps)
    return c


@pp.cell
def cutback_bend180(
    bend180=bend_euler180,
    straight_length=5.0,
    n_steps=6,
    cols=6,
    spacing=3,
    waveguide_factory=waveguide,
):
    """

    .. code::

          _
        _| |_| this is a stair

        _ this is a step

    """
    bend180 = pp.call_if_func(bend180)

    wg = waveguide_factory(length=straight_length, width=bend180.ports["W0"].width)
    wg_vertical = waveguide_factory(
        length=2 * bend180.size_info.width + straight_length + spacing,
        width=bend180.ports["W0"].width,
    )

    # Define a map between symbols and (component, input port, output port)
    string_to_device_in_out_ports = {
        "D": (bend180, "W0", "W1"),
        "C": (bend180, "W1", "W0"),
        "-": (wg, "W0", "E0"),
        "|": (wg_vertical, "W0", "E0"),
    }

    # Generate the sequence of staircases
    s = ""
    for i in range(cols):
        if i % 2 == 0:  # even row
            s += "D-C-" * n_steps + "|"
        else:
            s += "C-D-" * n_steps + "|"
    s = s[:-1]

    # Create the component from the sequence
    c = component_sequence(s, string_to_device_in_out_ports, start_orientation=0)
    c.update_settings(n_bends=n_steps * cols * 2 + cols * 2 - 2)
    return c


@pp.cell
def cutback_bend_circular(bend_radius=10.0, n_steps=3, n_stairs=4):
    bend90 = bend_circular(radius=bend_radius)
    c = cutback_bend(bend90=bend90, n_steps=n_steps, n_stairs=n_stairs)
    cc = add_fiber_array(c, optical_routing_type=1)
    return cc


if __name__ == "__main__":
    # c = cutback_bend_circular(n_steps=7, n_stairs=4, bend_radius=5) #62
    # c = cutback_bend_circular(n_steps=14, n_stairs=4) #118
    c = cutback_bend_circular(n_steps=3, n_stairs=4)  # 30
    # c = cutback_bend180()
    # c = cutback_bend90()
    pp.show(c)
