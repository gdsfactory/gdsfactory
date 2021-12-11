import gdsfactory as gf


def test_get_ports() -> None:
    c = gf.components.mzi_phase_shifter_top_heater_metal(length_x=123)

    p = c.get_ports_dict()
    assert len(p) == 4, len(p)

    p_dc = c.get_ports_dict(width=11.0)
    p_dc_layer = c.get_ports_dict(layer=(49, 0))
    assert len(p_dc) == 2, f"{len(p_dc)}"
    assert len(p_dc_layer) == 2, f"{len(p_dc_layer)}"

    p_optical = c.get_ports_dict(width=0.5)
    assert len(p_optical) == 2, f"{len(p_optical)}"

    p_optical_west = c.get_ports_dict(orientation=180, width=0.5)
    p_optical_east = c.get_ports_dict(orientation=0, width=0.5)
    assert len(p_optical_east) == 1, f"{len(p_optical_east)}"
    assert len(p_optical_west) == 1, f"{len(p_optical_west)}"


if __name__ == "__main__":
    test_get_ports()

    # c = gf.components.mzi_phase_shifter()
    # c.show()
    # p = c.get_ports_dict()
    # assert len(p) == 4, len(p)

    # p_dc = c.get_ports_dict(width=11.)
    # p_dc_layer = c.get_ports_dict(layer=(49, 0))
    # assert len(p_dc) == 2, f"{len(p_dc)}"
    # assert len(p_dc_layer) == 2, f"{len(p_dc_layer)}"

    # p_optical = c.get_ports_dict(width=0.5)
    # assert len(p_optical) == 2, f"{len(p_optical)}"

    # p_optical_west = c.get_ports_dict(orientation=180, width=0.5)
    # p_optical_east = c.get_ports_dict(orientation=0, width=0.5)
    # assert len(p_optical_east) == 1, f"{len(p_optical_east)}"
    # assert len(p_optical_west) == 1, f"{len(p_optical_west)}"
