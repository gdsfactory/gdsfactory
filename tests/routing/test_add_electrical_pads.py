import gdsfactory as gf


def test_add_electrical_pads_top() -> None:
    ring = gf.components.ring_single_heater(gap=0.2, radius=10, length_x=4)
    ring_with_grating_couplers = gf.routing.add_fiber_array(ring)
    c = gf.routing.add_electrical_pads_top(
        ring_with_grating_couplers, port_names=["l_e1", "r_e3"]
    )
    assert c
