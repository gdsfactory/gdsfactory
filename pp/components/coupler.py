import pp
from pp.components.coupler_symmetric import coupler_symmetric
from pp.components.coupler_symmetric import coupler_symmetric_biased
from pp.components.coupler_straight import coupler_straight
from pp.components.coupler_straight import coupler_straight_biased
from pp.netlist_to_gds import netlist_to_component
from pp.name import autoname
from pp.drc import assert_on_2nm_grid
from pp.drc import assert_on_1nm_grid


@autoname
def coupler(**kwargs):
    """ symmetric coupler

    Args:
        gap
        length
        coupler_symmetric_factory
        coupler_straight

    .. plot::
      :include-source:

      import pp

      c = pp.c.coupler(gap=0.2, length=10)
      pp.plotgds(c)

    """
    components, connections, ports_map = coupler_netlist(**kwargs)
    component = netlist_to_component(components, connections, ports_map)
    return component


@autoname
def coupler_biased(**kwargs):
    """ symmetric coupler
    """
    components, connections, ports_map = coupler_netlist(
        coupler_symmetric_factory=coupler_symmetric_biased,
        coupler_straight=coupler_straight_biased,
        **kwargs
    )
    component = netlist_to_component(components, connections, ports_map)
    return component


def coupler_netlist(
    wg_width=0.5,
    gap=0.236,
    length=20.007,
    coupler_symmetric_factory=coupler_symmetric,
    coupler_straight=coupler_straight,
):
    """
     SBEND_L-CS-SBEND_R
    """

    assert_on_1nm_grid(length)
    assert_on_2nm_grid(gap)

    _sbend = coupler_symmetric_factory(gap=gap, wg_width=wg_width)
    _cpl_straight = coupler_straight(length=length, gap=gap, width=wg_width)

    components = {
        "SBEND_L": (_sbend, "mirror_y"),
        "SBEND_R": (_sbend, "None"),
        "CS": (_cpl_straight, "None"),
    }

    connections = [("SBEND_L", "W0", "CS", "W0"), ("CS", "E0", "SBEND_R", "W0")]

    ports_map = {
        "W0": ("SBEND_L", "E0"),
        "W1": ("SBEND_L", "E1"),
        "E0": ("SBEND_R", "E0"),
        "E1": ("SBEND_R", "E1"),
    }

    return components, connections, ports_map


if __name__ == "__main__":
    from pp.routing import add_io_optical

    # c = coupler(gap=0.245, length=5.67, wg_width=0.2)
    # c = coupler(gap=0.2, length=5, wg_width=0.4)
    # c = coupler_biased(gap=0.2, length=5, wg_width=0.5)
    c = coupler()
    cc = add_io_optical(c)
    pp.show(cc)

    # from pp.routing.connect_component import add_io_optical
    # cc = add_io_optical(c)
    # pp.show(cc)
