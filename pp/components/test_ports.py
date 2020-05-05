# this code has been automatically generated from pp/components/__init__.py
import pp


def test_ring_double_bus(num_regression):
    c = pp.c.ring_double_bus()
    num_regression.check(c.get_ports_array())


def test_bend_euler180(num_regression):
    c = pp.c.bend_euler180()
    num_regression.check(c.get_ports_array())


def test_grating_coupler_elliptical_te(num_regression):
    c = pp.c.grating_coupler_elliptical_te()
    num_regression.check(c.get_ports_array())


def test_coupler(num_regression):
    c = pp.c.coupler()
    num_regression.check(c.get_ports_array())


def test_coupler_symmetric(num_regression):
    c = pp.c.coupler_symmetric()
    num_regression.check(c.get_ports_array())


def test_bend_circular_heater(num_regression):
    c = pp.c.bend_circular_heater()
    num_regression.check(c.get_ports_array())


def test_bend_circular180(num_regression):
    c = pp.c.bend_circular180()
    num_regression.check(c.get_ports_array())


def test_mmi2x2(num_regression):
    c = pp.c.mmi2x2()
    num_regression.check(c.get_ports_array())


def test_mzi1x2(num_regression):
    c = pp.c.mzi1x2()
    num_regression.check(c.get_ports_array())


def test_pad(num_regression):
    c = pp.c.pad()
    num_regression.check(c.get_ports_array())


def test_waveguide_heater(num_regression):
    c = pp.c.waveguide_heater()
    num_regression.check(c.get_ports_array())


def test_coupler_asymmetric(num_regression):
    c = pp.c.coupler_asymmetric()
    num_regression.check(c.get_ports_array())


def test_grating_coupler_uniform(num_regression):
    c = pp.c.grating_coupler_uniform()
    num_regression.check(c.get_ports_array())


def test_coupler90(num_regression):
    c = pp.c.coupler90()
    num_regression.check(c.get_ports_array())


def test_coupler_straight(num_regression):
    c = pp.c.coupler_straight()
    num_regression.check(c.get_ports_array())


def test_spiral_inner_io(num_regression):
    c = pp.c.spiral_inner_io()
    num_regression.check(c.get_ports_array())


def test_spiral_external_io(num_regression):
    c = pp.c.spiral_external_io()
    num_regression.check(c.get_ports_array())


def test_mmi1x2(num_regression):
    c = pp.c.mmi1x2()
    num_regression.check(c.get_ports_array())


def test_spiral_inner_io_euler(num_regression):
    c = pp.c.spiral_inner_io_euler()
    num_regression.check(c.get_ports_array())


def test_taper(num_regression):
    c = pp.c.taper()
    num_regression.check(c.get_ports_array())


def test_waveguide(num_regression):
    c = pp.c.waveguide()
    num_regression.check(c.get_ports_array())


def test_bend_circular(num_regression):
    c = pp.c.bend_circular()
    num_regression.check(c.get_ports_array())


def test_loop_mirror(num_regression):
    c = pp.c.loop_mirror()
    num_regression.check(c.get_ports_array())


def test_wg_heater_connected(num_regression):
    c = pp.c.wg_heater_connected()
    num_regression.check(c.get_ports_array())


def test_compass(num_regression):
    c = pp.c.compass()
    num_regression.check(c.get_ports_array())


def test_bend_s(num_regression):
    c = pp.c.bend_s()
    num_regression.check(c.get_ports_array())


def test_grating_coupler_elliptical_tm(num_regression):
    c = pp.c.grating_coupler_elliptical_tm()
    num_regression.check(c.get_ports_array())


def test_pad_array(num_regression):
    c = pp.c.pad_array()
    num_regression.check(c.get_ports_array())


def test_bend_euler90(num_regression):
    c = pp.c.bend_euler90()
    num_regression.check(c.get_ports_array())


def test_ring_single_bus(num_regression):
    c = pp.c.ring_single_bus()
    num_regression.check(c.get_ports_array())


def test_mzi2x2(num_regression):
    c = pp.c.mzi2x2()
    num_regression.check(c.get_ports_array())
