from lytest import contained_phidlDevice, difftest_it

import pp
from pp.test_containers import container_factory


@contained_phidlDevice
def C(top):
    # pp.clear_cache()
    top.add_ref(pp.c.C())


def test_gds_C():
    difftest_it(C)()


@contained_phidlDevice
def L(top):
    # pp.clear_cache()
    top.add_ref(pp.c.L())


def test_gds_L():
    difftest_it(L)()


@contained_phidlDevice
def bbox(top):
    # pp.clear_cache()
    top.add_ref(pp.c.bbox())


def test_gds_bbox():
    difftest_it(bbox)()


@contained_phidlDevice
def bend_circular(top):
    # pp.clear_cache()
    top.add_ref(pp.c.bend_circular())


def test_gds_bend_circular():
    difftest_it(bend_circular)()


@contained_phidlDevice
def bend_circular180(top):
    # pp.clear_cache()
    top.add_ref(pp.c.bend_circular180())


def test_gds_bend_circular180():
    difftest_it(bend_circular180)()


@contained_phidlDevice
def bend_circular_heater(top):
    # pp.clear_cache()
    top.add_ref(pp.c.bend_circular_heater())


def test_gds_bend_circular_heater():
    difftest_it(bend_circular_heater)()


@contained_phidlDevice
def bend_euler180(top):
    # pp.clear_cache()
    top.add_ref(pp.c.bend_euler180())


def test_gds_bend_euler180():
    difftest_it(bend_euler180)()


@contained_phidlDevice
def bend_euler90(top):
    # pp.clear_cache()
    top.add_ref(pp.c.bend_euler90())


def test_gds_bend_euler90():
    difftest_it(bend_euler90)()


@contained_phidlDevice
def bend_s(top):
    # pp.clear_cache()
    top.add_ref(pp.c.bend_s())


def test_gds_bend_s():
    difftest_it(bend_s)()


@contained_phidlDevice
def bezier(top):
    # pp.clear_cache()
    top.add_ref(pp.c.bezier())


def test_gds_bezier():
    difftest_it(bezier)()


@contained_phidlDevice
def cdc(top):
    # pp.clear_cache()
    top.add_ref(pp.c.cdc())


def test_gds_cdc():
    difftest_it(cdc)()


@contained_phidlDevice
def circle(top):
    # pp.clear_cache()
    top.add_ref(pp.c.circle())


def test_gds_circle():
    difftest_it(circle)()


@contained_phidlDevice
def compass(top):
    # pp.clear_cache()
    top.add_ref(pp.c.compass())


def test_gds_compass():
    difftest_it(compass)()


@contained_phidlDevice
def compensation_path(top):
    # pp.clear_cache()
    top.add_ref(pp.c.compensation_path())


def test_gds_compensation_path():
    difftest_it(compensation_path)()


@contained_phidlDevice
def component_lattice(top):
    # pp.clear_cache()
    top.add_ref(pp.c.component_lattice())


def test_gds_component_lattice():
    difftest_it(component_lattice)()


@contained_phidlDevice
def corner(top):
    # pp.clear_cache()
    top.add_ref(pp.c.corner())


def test_gds_corner():
    difftest_it(corner)()


@contained_phidlDevice
def coupler(top):
    # pp.clear_cache()
    top.add_ref(pp.c.coupler())


def test_gds_coupler():
    difftest_it(coupler)()


@contained_phidlDevice
def coupler90(top):
    # pp.clear_cache()
    top.add_ref(pp.c.coupler90())


def test_gds_coupler90():
    difftest_it(coupler90)()


@contained_phidlDevice
def coupler_adiabatic(top):
    # pp.clear_cache()
    top.add_ref(pp.c.coupler_adiabatic())


def test_gds_coupler_adiabatic():
    difftest_it(coupler_adiabatic)()


@contained_phidlDevice
def coupler_asymmetric(top):
    # pp.clear_cache()
    top.add_ref(pp.c.coupler_asymmetric())


def test_gds_coupler_asymmetric():
    difftest_it(coupler_asymmetric)()


@contained_phidlDevice
def coupler_full(top):
    # pp.clear_cache()
    top.add_ref(pp.c.coupler_full())


def test_gds_coupler_full():
    difftest_it(coupler_full)()


@contained_phidlDevice
def coupler_ring(top):
    # pp.clear_cache()
    top.add_ref(pp.c.coupler_ring())


def test_gds_coupler_ring():
    difftest_it(coupler_ring)()


@contained_phidlDevice
def coupler_straight(top):
    # pp.clear_cache()
    top.add_ref(pp.c.coupler_straight())


def test_gds_coupler_straight():
    difftest_it(coupler_straight)()


@contained_phidlDevice
def coupler_symmetric(top):
    # pp.clear_cache()
    top.add_ref(pp.c.coupler_symmetric())


def test_gds_coupler_symmetric():
    difftest_it(coupler_symmetric)()


@contained_phidlDevice
def cross(top):
    # pp.clear_cache()
    top.add_ref(pp.c.cross())


def test_gds_cross():
    difftest_it(cross)()


@contained_phidlDevice
def crossing(top):
    # pp.clear_cache()
    top.add_ref(pp.c.crossing())


def test_gds_crossing():
    difftest_it(crossing)()


@contained_phidlDevice
def crossing45(top):
    # pp.clear_cache()
    top.add_ref(pp.c.crossing45())


def test_gds_crossing45():
    difftest_it(crossing45)()


@contained_phidlDevice
def dbr(top):
    # pp.clear_cache()
    top.add_ref(pp.c.dbr())


def test_gds_dbr():
    difftest_it(dbr)()


@contained_phidlDevice
def dbr2(top):
    # pp.clear_cache()
    top.add_ref(pp.c.dbr2())


def test_gds_dbr2():
    difftest_it(dbr2)()


@contained_phidlDevice
def delay_snake(top):
    # pp.clear_cache()
    top.add_ref(pp.c.delay_snake())


def test_gds_delay_snake():
    difftest_it(delay_snake)()


@contained_phidlDevice
def disk(top):
    # pp.clear_cache()
    top.add_ref(pp.c.disk())


def test_gds_disk():
    difftest_it(disk)()


@contained_phidlDevice
def ellipse(top):
    # pp.clear_cache()
    top.add_ref(pp.c.ellipse())


def test_gds_ellipse():
    difftest_it(ellipse)()


@contained_phidlDevice
def grating_coupler_elliptical2(top):
    # pp.clear_cache()
    top.add_ref(pp.c.grating_coupler_elliptical2())


def test_gds_grating_coupler_elliptical2():
    difftest_it(grating_coupler_elliptical2)()


@contained_phidlDevice
def grating_coupler_elliptical_te(top):
    # pp.clear_cache()
    top.add_ref(pp.c.grating_coupler_elliptical_te())


def test_gds_grating_coupler_elliptical_te():
    difftest_it(grating_coupler_elliptical_te)()


@contained_phidlDevice
def grating_coupler_elliptical_tm(top):
    # pp.clear_cache()
    top.add_ref(pp.c.grating_coupler_elliptical_tm())


def test_gds_grating_coupler_elliptical_tm():
    difftest_it(grating_coupler_elliptical_tm)()


@contained_phidlDevice
def grating_coupler_te(top):
    # pp.clear_cache()
    top.add_ref(pp.c.grating_coupler_te())


def test_gds_grating_coupler_te():
    difftest_it(grating_coupler_te)()


@contained_phidlDevice
def grating_coupler_tm(top):
    # pp.clear_cache()
    top.add_ref(pp.c.grating_coupler_tm())


def test_gds_grating_coupler_tm():
    difftest_it(grating_coupler_tm)()


@contained_phidlDevice
def grating_coupler_tree(top):
    # pp.clear_cache()
    top.add_ref(pp.c.grating_coupler_tree())


def test_gds_grating_coupler_tree():
    difftest_it(grating_coupler_tree)()


@contained_phidlDevice
def grating_coupler_uniform(top):
    # pp.clear_cache()
    top.add_ref(pp.c.grating_coupler_uniform())


def test_gds_grating_coupler_uniform():
    difftest_it(grating_coupler_uniform)()


@contained_phidlDevice
def hline(top):
    # pp.clear_cache()
    top.add_ref(pp.c.hline())


def test_gds_hline():
    difftest_it(hline)()


@contained_phidlDevice
def litho_calipers(top):
    # pp.clear_cache()
    top.add_ref(pp.c.litho_calipers())


def test_gds_litho_calipers():
    difftest_it(litho_calipers)()


@contained_phidlDevice
def litho_star(top):
    # pp.clear_cache()
    top.add_ref(pp.c.litho_star())


def test_gds_litho_star():
    difftest_it(litho_star)()


@contained_phidlDevice
def litho_steps(top):
    # pp.clear_cache()
    top.add_ref(pp.c.litho_steps())


def test_gds_litho_steps():
    difftest_it(litho_steps)()


@contained_phidlDevice
def loop_mirror(top):
    # pp.clear_cache()
    top.add_ref(pp.c.loop_mirror())


def test_gds_loop_mirror():
    difftest_it(loop_mirror)()


@contained_phidlDevice
def mmi1x2(top):
    # pp.clear_cache()
    top.add_ref(pp.c.mmi1x2())


def test_gds_mmi1x2():
    difftest_it(mmi1x2)()


@contained_phidlDevice
def mmi2x2(top):
    # pp.clear_cache()
    top.add_ref(pp.c.mmi2x2())


def test_gds_mmi2x2():
    difftest_it(mmi2x2)()


@contained_phidlDevice
def mzi(top):
    # pp.clear_cache()
    top.add_ref(pp.c.mzi())


def test_gds_mzi():
    difftest_it(mzi)()


@contained_phidlDevice
def mzi1x2(top):
    # pp.clear_cache()
    top.add_ref(pp.c.mzi1x2())


def test_gds_mzi1x2():
    difftest_it(mzi1x2)()


@contained_phidlDevice
def mzi2x2(top):
    # pp.clear_cache()
    top.add_ref(pp.c.mzi2x2())


def test_gds_mzi2x2():
    difftest_it(mzi2x2)()


@contained_phidlDevice
def mzi_arm(top):
    # pp.clear_cache()
    top.add_ref(pp.c.mzi_arm())


def test_gds_mzi_arm():
    difftest_it(mzi_arm)()


@contained_phidlDevice
def mzi_lattice(top):
    # pp.clear_cache()
    top.add_ref(pp.c.mzi_lattice())


def test_gds_mzi_lattice():
    difftest_it(mzi_lattice)()


@contained_phidlDevice
def mzit(top):
    # pp.clear_cache()
    top.add_ref(pp.c.mzit())


def test_gds_mzit():
    difftest_it(mzit)()


@contained_phidlDevice
def mzit_lattice(top):
    # pp.clear_cache()
    top.add_ref(pp.c.mzit_lattice())


def test_gds_mzit_lattice():
    difftest_it(mzit_lattice)()


@contained_phidlDevice
def nxn(top):
    # pp.clear_cache()
    top.add_ref(pp.c.nxn())


def test_gds_nxn():
    difftest_it(nxn)()


@contained_phidlDevice
def pad(top):
    # pp.clear_cache()
    top.add_ref(pp.c.pad())


def test_gds_pad():
    difftest_it(pad)()


@contained_phidlDevice
def pad_array(top):
    # pp.clear_cache()
    top.add_ref(pp.c.pad_array())


def test_gds_pad_array():
    difftest_it(pad_array)()


@contained_phidlDevice
def pads_shorted(top):
    # pp.clear_cache()
    top.add_ref(pp.c.pads_shorted())


def test_gds_pads_shorted():
    difftest_it(pads_shorted)()


@contained_phidlDevice
def rectangle(top):
    # pp.clear_cache()
    top.add_ref(pp.c.rectangle())


def test_gds_rectangle():
    difftest_it(rectangle)()


@contained_phidlDevice
def ring(top):
    # pp.clear_cache()
    top.add_ref(pp.c.ring())


def test_gds_ring():
    difftest_it(ring)()


@contained_phidlDevice
def ring_double(top):
    # pp.clear_cache()
    top.add_ref(pp.c.ring_double())


def test_gds_ring_double():
    difftest_it(ring_double)()


@contained_phidlDevice
def ring_double_bus(top):
    # pp.clear_cache()
    top.add_ref(pp.c.ring_double_bus())


def test_gds_ring_double_bus():
    difftest_it(ring_double_bus)()


@contained_phidlDevice
def ring_single(top):
    # pp.clear_cache()
    top.add_ref(pp.c.ring_single())


def test_gds_ring_single():
    difftest_it(ring_single)()


@contained_phidlDevice
def ring_single_array(top):
    # pp.clear_cache()
    top.add_ref(pp.c.ring_single_array())


def test_gds_ring_single_array():
    difftest_it(ring_single_array)()


@contained_phidlDevice
def ring_single_bus(top):
    # pp.clear_cache()
    top.add_ref(pp.c.ring_single_bus())


def test_gds_ring_single_bus():
    difftest_it(ring_single_bus)()


@contained_phidlDevice
def spiral(top):
    # pp.clear_cache()
    top.add_ref(pp.c.spiral())


def test_gds_spiral():
    difftest_it(spiral)()


@contained_phidlDevice
def spiral_circular(top):
    # pp.clear_cache()
    top.add_ref(pp.c.spiral_circular())


def test_gds_spiral_circular():
    difftest_it(spiral_circular)()


@contained_phidlDevice
def spiral_external_io(top):
    # pp.clear_cache()
    top.add_ref(pp.c.spiral_external_io())


def test_gds_spiral_external_io():
    difftest_it(spiral_external_io)()


@contained_phidlDevice
def spiral_inner_io(top):
    # pp.clear_cache()
    top.add_ref(pp.c.spiral_inner_io())


def test_gds_spiral_inner_io():
    difftest_it(spiral_inner_io)()


@contained_phidlDevice
def spiral_inner_io_euler(top):
    # pp.clear_cache()
    top.add_ref(pp.c.spiral_inner_io_euler())


def test_gds_spiral_inner_io_euler():
    difftest_it(spiral_inner_io_euler)()


@contained_phidlDevice
def splitter_chain(top):
    # pp.clear_cache()
    top.add_ref(pp.c.splitter_chain())


def test_gds_splitter_chain():
    difftest_it(splitter_chain)()


@contained_phidlDevice
def splitter_tree(top):
    # pp.clear_cache()
    top.add_ref(pp.c.splitter_tree())


def test_gds_splitter_tree():
    difftest_it(splitter_tree)()


@contained_phidlDevice
def taper(top):
    # pp.clear_cache()
    top.add_ref(pp.c.taper())


def test_gds_taper():
    difftest_it(taper)()


@contained_phidlDevice
def taper_strip_to_ridge(top):
    # pp.clear_cache()
    top.add_ref(pp.c.taper_strip_to_ridge())


def test_gds_taper_strip_to_ridge():
    difftest_it(taper_strip_to_ridge)()


@contained_phidlDevice
def tlm(top):
    # pp.clear_cache()
    top.add_ref(pp.c.tlm())


def test_gds_tlm():
    difftest_it(tlm)()


@contained_phidlDevice
def verniers(top):
    # pp.clear_cache()
    top.add_ref(pp.c.verniers())


def test_gds_verniers():
    difftest_it(verniers)()


@contained_phidlDevice
def via(top):
    # pp.clear_cache()
    top.add_ref(pp.c.via())


def test_gds_via():
    difftest_it(via)()


@contained_phidlDevice
def via1(top):
    # pp.clear_cache()
    top.add_ref(pp.c.via1())


def test_gds_via1():
    difftest_it(via1)()


@contained_phidlDevice
def via2(top):
    # pp.clear_cache()
    top.add_ref(pp.c.via2())


def test_gds_via2():
    difftest_it(via2)()


@contained_phidlDevice
def via3(top):
    # pp.clear_cache()
    top.add_ref(pp.c.via3())


def test_gds_via3():
    difftest_it(via3)()


@contained_phidlDevice
def waveguide(top):
    # pp.clear_cache()
    top.add_ref(pp.c.waveguide())


def test_gds_waveguide():
    difftest_it(waveguide)()


@contained_phidlDevice
def waveguide_array(top):
    # pp.clear_cache()
    top.add_ref(pp.c.waveguide_array())


def test_gds_waveguide_array():
    difftest_it(waveguide_array)()


@contained_phidlDevice
def waveguide_heater(top):
    # pp.clear_cache()
    top.add_ref(pp.c.waveguide_heater())


def test_gds_waveguide_heater():
    difftest_it(waveguide_heater)()


@contained_phidlDevice
def waveguide_pin(top):
    # pp.clear_cache()
    top.add_ref(pp.c.waveguide_pin())


def test_gds_waveguide_pin():
    difftest_it(waveguide_pin)()


@contained_phidlDevice
def wg_heater_connected(top):
    # pp.clear_cache()
    top.add_ref(pp.c.wg_heater_connected())


def test_gds_wg_heater_connected():
    difftest_it(wg_heater_connected)()


@contained_phidlDevice
def wire(top):
    # pp.clear_cache()
    top.add_ref(pp.c.wire())


def test_gds_wire():
    difftest_it(wire)()


@contained_phidlDevice
def add_electrical_pads(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_electrical_pads"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_electrical_pads():
    difftest_it(add_electrical_pads)()


@contained_phidlDevice
def add_electrical_pads_shortest(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_electrical_pads_shortest"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_electrical_pads_shortest():
    difftest_it(add_electrical_pads_shortest)()


@contained_phidlDevice
def add_electrical_pads_top(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_electrical_pads_top"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_electrical_pads_top():
    difftest_it(add_electrical_pads_top)()


@contained_phidlDevice
def add_fiber_array(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_fiber_array"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_fiber_array():
    difftest_it(add_fiber_array)()


@contained_phidlDevice
def add_fiber_single(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_fiber_single"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_fiber_single():
    difftest_it(add_fiber_single)()


@contained_phidlDevice
def add_grating_couplers(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_grating_couplers"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_grating_couplers():
    difftest_it(add_grating_couplers)()


@contained_phidlDevice
def add_padding(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_padding"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_padding():
    difftest_it(add_padding)()


@contained_phidlDevice
def add_tapers(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_tapers"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_tapers():
    difftest_it(add_tapers)()


@contained_phidlDevice
def add_termination(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_termination"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_termination():
    difftest_it(add_termination)()


@contained_phidlDevice
def add_pins(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["add_pins"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_add_pins():
    difftest_it(add_pins)()


@contained_phidlDevice
def extend_ports(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["extend_ports"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_extend_ports():
    difftest_it(extend_ports)()


@contained_phidlDevice
def package_optical2x2(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["package_optical2x2"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_package_optical2x2():
    difftest_it(package_optical2x2)()


@contained_phidlDevice
def rotate(top):
    # pp.clear_cache()
    component = pp.c.mzi2x2(with_elec_connections=True)
    container_function = container_factory["rotate"]
    container = container_function(component=component)
    top.add_ref(container)


def test_gds_rotate():
    difftest_it(rotate)()
