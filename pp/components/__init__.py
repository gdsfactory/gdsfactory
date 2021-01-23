""" NOTE: import order matters.
Only change the order if you know what you are doing

isort:skip_file
"""
# level 0 components
from pp.components.waveguide import waveguide
from pp.components.waveguide_heater import waveguide_heater
from pp.components.waveguide_heater import wg_heater_connected
from pp.components.waveguide_pin import waveguide_pin
from pp.components.waveguide_array import waveguide_array

from pp.components.bend_circular import bend_circular
from pp.components.bend_circular import bend_circular180
from pp.components.bend_circular_heater import bend_circular_heater
from pp.components.bend_s import bend_s
from pp.components.bezier import bezier
from pp.components.euler.bend_euler import bend_euler90
from pp.components.euler.bend_euler import bend_euler180

from pp.components.coupler90 import coupler90
from pp.components.coupler_straight import coupler_straight
from pp.components.coupler_symmetric import coupler_symmetric
from pp.components.coupler_asymmetric import coupler_asymmetric
from pp.components.hline import hline

# basic shapes
from pp.components.circle import circle
from pp.components.compass import compass
from pp.components.cross import cross
from pp.components.crossing_waveguide import crossing
from pp.components.crossing_waveguide import crossing45
from pp.components.crossing_waveguide import compensation_path
from pp.components.ellipse import ellipse
from pp.components.label import label
from pp.components.rectangle import rectangle
from pp.components.ring import ring
from pp.components.taper import taper
from pp.components.taper import taper_strip_to_ridge
from pp.components.taper_from_csv import taper_0p5_to_3_l36
from pp.components.text import text
from pp.components.L import L
from pp.components.C import C
from pp.components.bbox import bbox
from pp.components.nxn import nxn

# optical test structures
from pp.components.version_stamp import version_stamp
from pp.components.version_stamp import qrcode
from pp.components.manhattan_font import manhattan_text
from pp.components.logo import logo
from pp.components.align import align_wafer
from pp.components.cutback_bend import cutback_bend90
from pp.components.cutback_bend import cutback_bend180
from pp.components.cutback_component import cutback_component
from pp.components.cutback_component import cutback_component_flipped


from pp.components.pcm.litho_calipers import litho_calipers
from pp.components.pcm.litho_star import litho_star
from pp.components.pcm.litho_steps import litho_steps
from pp.components.pcm.verniers import verniers

from pp.components.grating_coupler.elliptical import grating_coupler_elliptical_te
from pp.components.grating_coupler.elliptical import grating_coupler_elliptical_tm
from pp.components.grating_coupler.elliptical2 import grating_coupler_elliptical2
from pp.components.grating_coupler.uniform import grating_coupler_uniform
from pp.components.grating_coupler.uniform_optimized import (
    grating_coupler_uniform_optimized,
)
from pp.components.grating_coupler.grating_coupler_tree import grating_coupler_tree
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_te
from pp.components.grating_coupler.elliptical_trenches import grating_coupler_tm
from pp.components.grating_coupler.grating_coupler_loss import grating_coupler_loss
from pp.components.delay_snake import delay_snake
from pp.components.spiral import spiral
from pp.components.spiral_inner_io import spiral_inner_io_euler
from pp.components.spiral_inner_io import spiral_inner_io
from pp.components.spiral_external_io import spiral_external_io
from pp.components.spiral_circular import spiral_circular
from pp.components.cdc import cdc
from pp.components.dbr import dbr
from pp.components.dbr2 import dbr2

# electrical
from pp.components.electrical.wire import wire
from pp.components.electrical.wire import corner
from pp.components.electrical.pad import pad
from pp.components.electrical.pad import pad_array
from pp.components.electrical.tlm import via
from pp.components.electrical.tlm import via1
from pp.components.electrical.tlm import via2
from pp.components.electrical.tlm import via3
from pp.components.electrical.tlm import tlm
from pp.components.electrical.pads_shorted import pads_shorted

# electrical PCM
from pp.components.pcm.test_resistance import test_resistance
from pp.components.pcm.test_via import test_via

# level 1 components
from pp.components.cavity import cavity
from pp.components.coupler import coupler
from pp.components.coupler_ring import coupler_ring
from pp.components.coupler_adiabatic import coupler_adiabatic
from pp.components.coupler_full import coupler_full
from pp.components.disk import disk
from pp.components.ring_single import ring_single
from pp.components.ring_single_array import ring_single_array
from pp.components.ring_double import ring_double
from pp.components.ring_single_bus import ring_single_bus
from pp.components.ring_double_bus import ring_double_bus
from pp.components.mmi1x2 import mmi1x2
from pp.components.mmi2x2 import mmi2x2
from pp.components.mzi2x2 import mzi_arm
from pp.components.mzi2x2 import mzi2x2
from pp.components.mzi1x2 import mzi1x2
from pp.components.mzi import mzi
from pp.components.mzit import mzit
from pp.components.mzi_lattice import mzi_lattice
from pp.components.mzit_lattice import mzit_lattice
from pp.components.loop_mirror import loop_mirror

# level 2 components
from pp.components.component_lattice import component_lattice
from pp.components.component_sequence import component_sequence
from pp.components.splitter_tree import splitter_tree
from pp.components.splitter_chain import splitter_chain


# we will test each factory component hash, ports and properties """
component_factory = dict(
    align_wafer=align_wafer,
    bend_circular180=bend_circular180,
    bend_circular=bend_circular,
    bend_circular_heater=bend_circular_heater,
    bend_euler180=bend_euler180,
    bend_euler90=bend_euler90,
    bend_s=bend_s,
    bezier=bezier,
    cavity=cavity,
    cdc=cdc,
    circle=circle,
    compass=compass,
    compensation_path=compensation_path,
    component_lattice=component_lattice,
    component_sequence=component_sequence,
    corner=corner,
    coupler90=coupler90,
    coupler=coupler,
    coupler_adiabatic=coupler_adiabatic,
    coupler_asymmetric=coupler_asymmetric,
    coupler_full=coupler_full,
    coupler_ring=coupler_ring,
    coupler_straight=coupler_straight,
    coupler_symmetric=coupler_symmetric,
    cross=cross,
    crossing45=crossing45,
    crossing=crossing,
    cutback_bend90=cutback_bend90,
    cutback_bend180=cutback_bend180,
    cutback_component=cutback_component,
    cutback_component_flipped=cutback_component_flipped,
    dbr2=dbr2,
    dbr=dbr,
    delay_snake=delay_snake,
    disk=disk,
    ellipse=ellipse,
    nxn=nxn,
    grating_coupler_elliptical2=grating_coupler_elliptical2,
    grating_coupler_elliptical_te=grating_coupler_elliptical_te,
    grating_coupler_elliptical_tm=grating_coupler_elliptical_tm,
    grating_coupler_te=grating_coupler_te,
    grating_coupler_tm=grating_coupler_tm,
    grating_coupler_loss=grating_coupler_loss,
    grating_coupler_tree=grating_coupler_tree,
    grating_coupler_uniform=grating_coupler_uniform,
    grating_coupler_uniform_optimized=grating_coupler_uniform_optimized,
    hline=hline,
    label=label,
    litho_calipers=litho_calipers,
    litho_star=litho_star,
    litho_steps=litho_steps,
    loop_mirror=loop_mirror,
    mmi1x2=mmi1x2,
    mmi2x2=mmi2x2,
    mzi1x2=mzi1x2,
    mzi2x2=mzi2x2,
    mzi=mzi,
    mzit=mzit,
    mzi_lattice=mzi_lattice,
    mzit_lattice=mzit_lattice,
    mzi_arm=mzi_arm,
    pad=pad,
    pad_array=pad_array,
    pads_shorted=pads_shorted,
    rectangle=rectangle,
    ring=ring,
    ring_double=ring_double,
    ring_double_bus=ring_double_bus,
    ring_single=ring_single,
    ring_single_array=ring_single_array,
    ring_single_bus=ring_single_bus,
    spiral=spiral,
    spiral_circular=spiral_circular,
    spiral_external_io=spiral_external_io,
    spiral_inner_io=spiral_inner_io,
    spiral_inner_io_euler=spiral_inner_io_euler,
    splitter_chain=splitter_chain,
    splitter_tree=splitter_tree,
    taper=taper,
    taper_0p5_to_3_l36=taper_0p5_to_3_l36,
    taper_strip_to_ridge=taper_strip_to_ridge,
    test_resistance=test_resistance,
    test_via=test_via,
    text=text,
    tlm=tlm,
    verniers=verniers,
    via1=via1,
    via2=via2,
    via3=via3,
    via=via,
    manhattan_text=manhattan_text,
    qrcode=qrcode,
    version_stamp=version_stamp,
    logo=logo,
    waveguide=waveguide,
    waveguide_array=waveguide_array,
    waveguide_heater=waveguide_heater,
    waveguide_pin=waveguide_pin,
    wg_heater_connected=wg_heater_connected,
    wire=wire,
    C=C,
    L=L,
    bbox=bbox,
)


def factory(component_type, component_factory=component_factory, **settings):
    """Returns a component with settings."""
    import pp

    if isinstance(component_type, pp.Component):
        return component_type
    elif callable(component_type):
        return component_type(**settings)
    elif component_type not in component_factory.keys():
        raise ValueError(
            f"component_type = {component_type} not in: \n"
            + "\n".join(component_factory.keys())
        )
    return component_factory[component_type](**settings)


component_names_skip_test = [
    "label",
    "text",
    "component_sequence",
    "compensation_path",
    "component_lattice",
    "version_stamp",
]
component_names_skip_test_ports = ["coupler"]

container_names = ["cavity"]
component_names = (
    set(component_factory.keys())
    - set(component_names_skip_test)
    - set(container_names)
)
component_names_test_ports = component_names - set(component_names_skip_test_ports)
circuit_names = {
    "mzi",
    "ring_single",
    "ring_single_array",
    "ring_double",
    "mzit_lattice",
    "mzit",
    "component_lattice",
}

__all__ = list(component_factory.keys())

if __name__ == "__main__":
    for c in component_names:
        ci = component_factory[c]()
