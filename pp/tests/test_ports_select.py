import pp


def test_get_ports():
    c = pp.components.mzi2x2(with_elec_connections=True)

    p = c.get_ports_dict()
    assert len(p) == 7

    p_dc = c.get_ports_dict(port_type="dc")
    p_dc_layer = c.get_ports_dict(layer=(49, 0))
    assert len(p_dc) == 3
    assert len(p_dc_layer) == 3

    p_optical = c.get_ports_dict(port_type="optical")
    assert len(p_optical) == 4

    p_optical_west = c.get_ports_dict(prefix="W")
    p_optical_east = c.get_ports_dict(prefix="E")
    assert len(p_optical_east) == 2
    assert len(p_optical_west) == 2


if __name__ == "__main__":
    test_get_ports()
    # c = pp.components.mzi2x2(with_elec_connections=True)
    # p_dc_layer = c.get_ports_dict(layer=(49, 0))
    # p_dc = c.get_ports_dict(port_type="dc")

    # print('p_dc', p_dc.keys())
    # print('p_dc_layer', p_dc_layer.keys())

    # p = c.get_ports_dict()
    # p_optical = c.get_ports_dict(port_type="optical")
    # p_optical_west = c.get_ports_dict(prefix="W")
    # p_optical_east = c.get_ports_dict(prefix="E")
