# this code has been automatically generated from pp/components/write_tests.py
import pp


def test_grating_coupler_tree(num_regression):
    c = pp.c.grating_coupler_tree()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_ellipse(num_regression):
    c = pp.c.ellipse()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_bend_s(num_regression):
    c = pp.c.bend_s()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_compass(num_regression):
    c = pp.c.compass()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_grating_coupler_elliptical_te(num_regression):
    c = pp.c.grating_coupler_elliptical_te()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_crossing(num_regression):
    c = pp.c.crossing()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_mzi(num_regression):
    c = pp.c.mzi()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_rectangle(num_regression):
    c = pp.c.rectangle()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_hline(num_regression):
    c = pp.c.hline()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_waveguide_heater(num_regression):
    c = pp.c.waveguide_heater()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_loop_mirror(num_regression):
    c = pp.c.loop_mirror()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_circle(num_regression):
    c = pp.c.circle()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_litho_star(num_regression):
    c = pp.c.litho_star()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_litho_steps(num_regression):
    c = pp.c.litho_steps()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_pad_array(num_regression):
    c = pp.c.pad_array()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_grating_coupler_te(num_regression):
    c = pp.c.grating_coupler_te()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_ring_double_bus(num_regression):
    c = pp.c.ring_double_bus()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_spiral_inner_io(num_regression):
    c = pp.c.spiral_inner_io()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_mmi1x2(num_regression):
    c = pp.c.mmi1x2()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_ring(num_regression):
    c = pp.c.ring()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_coupler(num_regression):
    c = pp.c.coupler()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_taper(num_regression):
    c = pp.c.taper()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_ring_single(num_regression):
    c = pp.c.ring_single()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_delay_snake(num_regression):
    c = pp.c.delay_snake()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_bend_circular180(num_regression):
    c = pp.c.bend_circular180()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_coupler_ring(num_regression):
    c = pp.c.coupler_ring()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_mmi2x2(num_regression):
    c = pp.c.mmi2x2()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_coupler_symmetric(num_regression):
    c = pp.c.coupler_symmetric()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_mzi_arm(num_regression):
    c = pp.c.mzi_arm()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_text(num_regression):
    c = pp.c.text()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_waveguide(num_regression):
    c = pp.c.waveguide()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_bend_circular_heater(num_regression):
    c = pp.c.bend_circular_heater()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_grating_coupler_tm(num_regression):
    c = pp.c.grating_coupler_tm()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_wg_heater_connected(num_regression):
    c = pp.c.wg_heater_connected()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_bend_circular(num_regression):
    c = pp.c.bend_circular()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_litho_calipers(num_regression):
    c = pp.c.litho_calipers()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_coupler_asymmetric(num_regression):
    c = pp.c.coupler_asymmetric()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_coupler_straight(num_regression):
    c = pp.c.coupler_straight()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_rectangle_centered(num_regression):
    c = pp.c.rectangle_centered()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_test_via(num_regression):
    c = pp.c.test_via()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_bend_euler180(num_regression):
    c = pp.c.bend_euler180()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_test_resistance(num_regression):
    c = pp.c.test_resistance()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_grating_coupler_uniform(num_regression):
    c = pp.c.grating_coupler_uniform()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_component_lattice(num_regression):
    c = pp.c.component_lattice()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_coupler90(num_regression):
    c = pp.c.coupler90()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_dbr(num_regression):
    c = pp.c.dbr()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_spiral_inner_io_euler(num_regression):
    c = pp.c.spiral_inner_io_euler()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_ring_double(num_regression):
    c = pp.c.ring_double()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_pad(num_regression):
    c = pp.c.pad()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_cross(num_regression):
    c = pp.c.cross()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_spiral_external_io(num_regression):
    c = pp.c.spiral_external_io()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_taper_strip_to_ridge(num_regression):
    c = pp.c.taper_strip_to_ridge()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_bend_euler90(num_regression):
    c = pp.c.bend_euler90()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_ring_single_bus(num_regression):
    c = pp.c.ring_single_bus()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_bezier(num_regression):
    c = pp.c.bezier()
    if c.ports:
        num_regression.check(c.get_ports_array())


def test_grating_coupler_elliptical_tm(num_regression):
    c = pp.c.grating_coupler_elliptical_tm()
    if c.ports:
        num_regression.check(c.get_ports_array())
