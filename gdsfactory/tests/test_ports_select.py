import gdsfactory as gf


def test_get_ports() -> None:
    c = gf.components.mzi_phase_shifter()

    p = c.get_ports_dict()
    assert len(p) == 4, f"{len(p)}"

    p_dc = c.get_ports_dict(port_type="dc")
    p_dc_layer = c.get_ports_dict(layer=(49, 0))
    assert len(p_dc) == 2, f"{len(p_dc)}"
    assert len(p_dc_layer) == 2, f"{len(p_dc_layer)}"

    p_optical = c.get_ports_dict(port_type="optical")
    assert len(p_optical) == 2, f"{len(p_optical)}"

    p_optical_west = c.get_ports_dict(prefix="W")
    p_optical_east = c.get_ports_dict(prefix="E")
    assert len(p_optical_east) == 1, f"{len(p_optical_east)}"
    assert len(p_optical_west) == 1, f"{len(p_optical_west)}"


if __name__ == "__main__":
    test_get_ports()
    c = gf.components.mzi_phase_shifter()
    c.show()

    # p_dc_layer = c.get_ports_dict(layer=(49, 0))
    # p_dc = c.get_ports_dict(port_type="dc")

    # print('p_dc', p_dc.keys())
    # print('p_dc_layer', p_dc_layer.keys())

    # p = c.get_ports_dict()
    # p_optical = c.get_ports_dict(port_type="optical")
    # p_optical_west = c.get_ports_dict(prefix="W")
    # p_optical_east = c.get_ports_dict(prefix="E")
