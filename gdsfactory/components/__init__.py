from gdsfactory.components.align import add_frame, align_wafer, triangle
from gdsfactory.components.array import array
from gdsfactory.components.array_with_fanout import (
    array_with_fanout,
    array_with_fanout_2d,
)
from gdsfactory.components.array_with_via import array_with_via, array_with_via_2d
from gdsfactory.components.awg import awg
from gdsfactory.components.bbox import Coordinate, bbox
from gdsfactory.components.bend_circular import bend_circular, bend_circular180
from gdsfactory.components.bend_circular_heater import bend_circular_heater
from gdsfactory.components.bend_euler import bend_euler, bend_euler180, bend_euler_s
from gdsfactory.components.bend_port import bend_port
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.C import C
from gdsfactory.components.cavity import cavity
from gdsfactory.components.cdc import cdc
from gdsfactory.components.circle import circle
from gdsfactory.components.compass import compass
from gdsfactory.components.component_lattice import component_lattice
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.coupler import coupler
from gdsfactory.components.coupler90 import coupler90, coupler90circular
from gdsfactory.components.coupler90bend import coupler90bend
from gdsfactory.components.coupler_adiabatic import coupler_adiabatic
from gdsfactory.components.coupler_asymmetric import coupler_asymmetric
from gdsfactory.components.coupler_full import coupler_full
from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.coupler_straight import coupler_straight
from gdsfactory.components.coupler_symmetric import coupler_symmetric
from gdsfactory.components.cross import cross
from gdsfactory.components.crossing_waveguide import (
    compensation_path,
    crossing,
    crossing45,
    crossing_arm,
    crossing_etched,
    crossing_from_taper,
)
from gdsfactory.components.cutback_bend import (
    cutback_bend,
    cutback_bend90,
    cutback_bend90circular,
    cutback_bend180,
    cutback_bend180circular,
    staircase,
)
from gdsfactory.components.cutback_component import (
    cutback_component,
    cutback_component_flipped,
)
from gdsfactory.components.dbr import dbr
from gdsfactory.components.dbr2 import dbr2
from gdsfactory.components.delay_snake import delay_snake
from gdsfactory.components.delay_snake2 import delay_snake2, test_delay_snake2_length
from gdsfactory.components.delay_snake3 import delay_snake3, test_delay_snake3_length
from gdsfactory.components.die import die
from gdsfactory.components.die_bbox import big_square, die_bbox
from gdsfactory.components.disk import disk
from gdsfactory.components.ellipse import ellipse
from gdsfactory.components.extend_ports_list import extend_ports_list
from gdsfactory.components.extension import extend_port, extend_ports
from gdsfactory.components.fiber import fiber
from gdsfactory.components.fiber_array import fiber_array
from gdsfactory.components.grating_coupler_array import grating_coupler_array
from gdsfactory.components.grating_coupler_elliptical import (
    ellipse_arc,
    grating_coupler_elliptical,
    grating_coupler_elliptical_te,
    grating_coupler_elliptical_tm,
    grating_taper_points,
    grating_tooth_points,
)
from gdsfactory.components.grating_coupler_elliptical2 import (
    grating_coupler_elliptical2,
)
from gdsfactory.components.grating_coupler_elliptical_trenches import (
    grating_coupler_elliptical_trenches,
    grating_coupler_te,
    grating_coupler_tm,
)
from gdsfactory.components.grating_coupler_loss import (
    connect_loopback,
    grating_coupler_loss,
    loss_deembedding_ch12_34,
    loss_deembedding_ch13_24,
    loss_deembedding_ch14_23,
)
from gdsfactory.components.grating_coupler_tree import grating_coupler_tree
from gdsfactory.components.grating_coupler_uniform import grating_coupler_uniform
from gdsfactory.components.hline import hline
from gdsfactory.components.L import L
from gdsfactory.components.litho_calipers import litho_calipers
from gdsfactory.components.litho_ruler import litho_ruler
from gdsfactory.components.litho_steps import litho_steps
from gdsfactory.components.logo import logo
from gdsfactory.components.loop_mirror import loop_mirror
from gdsfactory.components.manhattan_font import manhattan_text
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.mmi2x2 import mmi2x2
from gdsfactory.components.mzi import mzi
from gdsfactory.components.mzi_arm import mzi_arm
from gdsfactory.components.mzi_lattice import mzi_lattice
from gdsfactory.components.mzi_phase_shifter import (
    mzi_phase_shifter,
    mzi_phase_shifter_90_90,
)
from gdsfactory.components.mzit import mzit
from gdsfactory.components.mzit_lattice import mzit_lattice
from gdsfactory.components.nxn import nxn
from gdsfactory.components.pad import pad, pad_array, pad_array90, pad_array270
from gdsfactory.components.pads_shorted import pads_shorted
from gdsfactory.components.pcm_optical import pcm_optical
from gdsfactory.components.ramp import ramp
from gdsfactory.components.rectangle import DIRECTION_TO_ANGLE, rectangle
from gdsfactory.components.resistance_meander import resistance_meander
from gdsfactory.components.ring import ring
from gdsfactory.components.ring_double import ring_double
from gdsfactory.components.ring_single import ring_single
from gdsfactory.components.ring_single_array import ring_single_array
from gdsfactory.components.ring_single_dut import ring_single_dut, taper2
from gdsfactory.components.spiral import spiral
from gdsfactory.components.spiral_circular import spiral_circular
from gdsfactory.components.spiral_external_io import spiral_external_io
from gdsfactory.components.spiral_inner_io import (
    spiral_inner_io,
    spiral_inner_io_fiber_single,
)
from gdsfactory.components.splitter_chain import splitter_chain
from gdsfactory.components.splitter_tree import (
    splitter_tree,
    test_splitter_tree_ports,
    test_splitter_tree_ports_no_sbend,
)
from gdsfactory.components.straight import straight
from gdsfactory.components.straight_array import straight_array
from gdsfactory.components.straight_heater_doped_rib import straight_heater_doped_rib
from gdsfactory.components.straight_heater_doped_strip import (
    straight_heater_doped_strip,
)
from gdsfactory.components.straight_heater_meander import straight_heater_meander
from gdsfactory.components.straight_heater_metal import (
    straight_heater_metal,
    straight_heater_metal_90_90,
    straight_heater_metal_undercut,
    straight_heater_metal_undercut_90_90,
    test_ports,
)
from gdsfactory.components.straight_pin import straight_pin, straight_pn
from gdsfactory.components.straight_pin_slot import straight_pin_slot
from gdsfactory.components.straight_rib import straight_rib, straight_rib_tapered
from gdsfactory.components.switch_tree import switch_tree
from gdsfactory.components.taper import (
    taper,
    taper_strip_to_ridge,
    taper_strip_to_ridge_trenches,
)
from gdsfactory.components.taper_cross_section import (
    taper_cross_section_linear,
    taper_cross_section_sine,
)
from gdsfactory.components.taper_from_csv import (
    taper_0p5_to_3_l36,
    taper_from_csv,
    taper_w10_l100,
    taper_w10_l150,
    taper_w10_l200,
    taper_w11_l200,
    taper_w12_l200,
)
from gdsfactory.components.text import githash, text
from gdsfactory.components.verniers import verniers
from gdsfactory.components.version_stamp import pixel, qrcode, version_stamp
from gdsfactory.components.via import via, via1, via2, viac
from gdsfactory.components.via_cutback import via_cutback
from gdsfactory.components.via_stack import via_stack, via_stack_heater, via_stack_slab
from gdsfactory.components.via_stack_slot import via_stack_slot
from gdsfactory.components.via_stack_with_offset import via_stack_with_offset
from gdsfactory.components.waveguide_template import strip
from gdsfactory.components.wire import wire_corner, wire_straight
from gdsfactory.components.wire_sbend import wire_sbend

__all__ = [
    "C",
    "Coordinate",
    "DIRECTION_TO_ANGLE",
    "L",
    "add_frame",
    "align",
    "align_wafer",
    "array",
    "array_with_fanout",
    "array_with_fanout_2d",
    "array_with_via",
    "array_with_via_2d",
    "awg",
    "bbox",
    "bend_circular",
    "bend_circular180",
    "bend_circular_heater",
    "bend_euler",
    "bend_euler180",
    "bend_euler_s",
    "bend_port",
    "bend_s",
    "big_square",
    "cavity",
    "circle",
    "compass",
    "compensation_path",
    "component_lattice",
    "component_sequence",
    "connect_loopback",
    "coupler",
    "coupler90",
    "coupler90bend",
    "coupler90circular",
    "coupler_adiabatic",
    "coupler_asymmetric",
    "coupler_full",
    "coupler_ring",
    "coupler_straight",
    "coupler_symmetric",
    "cross",
    "crossing",
    "crossing45",
    "crossing_arm",
    "crossing_etched",
    "crossing_from_taper",
    "crossing_waveguide",
    "cutback_bend",
    "cutback_bend180",
    "cutback_bend180circular",
    "cutback_bend90",
    "cutback_bend90circular",
    "cutback_component",
    "cutback_component_flipped",
    "dbr",
    "dbr2",
    "delay_snake",
    "delay_snake2",
    "delay_snake3",
    "die",
    "die_bbox",
    "disk",
    "ellipse",
    "ellipse_arc",
    "extend_port",
    "extend_ports",
    "extend_ports_list",
    "extension",
    "fiber",
    "fiber_array",
    "githash",
    "grating_coupler_array",
    "grating_coupler_elliptical",
    "grating_coupler_elliptical2",
    "grating_coupler_elliptical_te",
    "grating_coupler_elliptical_tm",
    "grating_coupler_elliptical_trenches",
    "grating_coupler_functions",
    "grating_coupler_loss",
    "grating_coupler_te",
    "grating_coupler_tm",
    "grating_coupler_tree",
    "grating_coupler_uniform",
    "grating_taper_points",
    "grating_tooth_points",
    "hline",
    "litho_calipers",
    "litho_ruler",
    "litho_steps",
    "logo",
    "loop_mirror",
    "loss_deembedding_ch12_34",
    "loss_deembedding_ch13_24",
    "loss_deembedding_ch14_23",
    "manhattan_font",
    "manhattan_text",
    "mmi1x2",
    "mmi2x2",
    "mzi",
    "mzi_arm",
    "mzi_lattice",
    "mzi_phase_shifter",
    "mzi_phase_shifter_90_90",
    "mzit",
    "mzit_lattice",
    "nxn",
    "pcm_optical",
    "pad",
    "pad_array",
    "pad_array90",
    "pad_array270",
    "pads_shorted",
    "pixel",
    "qrcode",
    "ramp",
    "rectangle",
    "resistance_meander",
    "ring",
    "ring_double",
    "ring_single",
    "ring_single_array",
    "ring_single_dut",
    "spiral",
    "spiral_circular",
    "spiral_external_io",
    "spiral_inner_io",
    "splitter_chain",
    "splitter_tree",
    "staircase",
    "straight",
    "straight_array",
    "straight_heater_doped_rib",
    "straight_heater_doped_strip",
    "straight_heater_metal",
    "straight_heater_metal_90_90",
    "straight_heater_metal_undercut",
    "straight_heater_metal_undercut_90_90",
    "straight_pin",
    "straight_pn",
    "straight_rib",
    "strip",
    "taper",
    "taper2",
    "taper_0p5_to_3_l36",
    "taper_from_csv",
    "taper_strip_to_ridge",
    "taper_strip_to_ridge_trenches",
    "taper_w10_l100",
    "taper_w10_l150",
    "taper_w10_l200",
    "taper_w11_l200",
    "taper_w12_l200",
    "test_delay_snake2_length",
    "test_delay_snake3_length",
    "test_ports",
    "test_splitter_tree_ports",
    "test_splitter_tree_ports_no_sbend",
    "text",
    "triangle",
    "verniers",
    "version_stamp",
    "via",
    "viac",
    "via1",
    "via2",
    "via_cutback",
    "via_stack",
    "via_stack_heater",
    "via_stack_slab",
    "via_stack_with_offset",
    "waveguide_template",
    "wire",
    "wire_corner",
    "wire_sbend",
    "wire_straight",
]

factory = dict(
    C=C,
    L=L,
    add_frame=add_frame,
    align_wafer=align_wafer,
    array=array,
    array_with_fanout=array_with_fanout,
    array_with_fanout_2d=array_with_fanout_2d,
    array_with_via=array_with_via,
    array_with_via_2d=array_with_via_2d,
    awg=awg,
    bbox=bbox,
    bend_circular=bend_circular,
    bend_circular180=bend_circular180,
    bend_circular_heater=bend_circular_heater,
    bend_euler=bend_euler,
    bend_euler180=bend_euler180,
    bend_euler_s=bend_euler_s,
    bend_port=bend_port,
    bend_s=bend_s,
    cavity=cavity,
    cdc=cdc,
    circle=circle,
    compass=compass,
    compensation_path=compensation_path,
    component_lattice=component_lattice,
    component_sequence=component_sequence,
    coupler=coupler,
    coupler90=coupler90,
    coupler90bend=coupler90bend,
    coupler90circular=coupler90circular,
    coupler_adiabatic=coupler_adiabatic,
    coupler_asymmetric=coupler_asymmetric,
    coupler_full=coupler_full,
    coupler_ring=coupler_ring,
    coupler_straight=coupler_straight,
    coupler_symmetric=coupler_symmetric,
    cross=cross,
    crossing=crossing,
    crossing45=crossing45,
    crossing_arm=crossing_arm,
    crossing_etched=crossing_etched,
    crossing_from_taper=crossing_from_taper,
    cutback_bend=cutback_bend,
    cutback_bend180=cutback_bend180,
    cutback_bend180circular=cutback_bend180circular,
    cutback_bend90=cutback_bend90,
    cutback_bend90circular=cutback_bend90circular,
    cutback_component=cutback_component,
    cutback_component_flipped=cutback_component_flipped,
    dbr=dbr,
    dbr2=dbr2,
    delay_snake=delay_snake,
    delay_snake2=delay_snake2,
    delay_snake3=delay_snake3,
    die=die,
    die_bbox=die_bbox,
    disk=disk,
    ellipse=ellipse,
    extend_port=extend_port,
    extend_ports=extend_ports,
    extend_ports_list=extend_ports_list,
    fiber=fiber,
    fiber_array=fiber_array,
    grating_coupler_array=grating_coupler_array,
    grating_coupler_elliptical=grating_coupler_elliptical,
    grating_coupler_elliptical2=grating_coupler_elliptical2,
    grating_coupler_elliptical_te=grating_coupler_elliptical_te,
    grating_coupler_elliptical_tm=grating_coupler_elliptical_tm,
    grating_coupler_elliptical_trenches=grating_coupler_elliptical_trenches,
    grating_coupler_loss=grating_coupler_loss,
    grating_coupler_te=grating_coupler_te,
    grating_coupler_tm=grating_coupler_tm,
    grating_coupler_tree=grating_coupler_tree,
    grating_coupler_uniform=grating_coupler_uniform,
    hline=hline,
    litho_calipers=litho_calipers,
    litho_steps=litho_steps,
    logo=logo,
    loop_mirror=loop_mirror,
    loss_deembedding_ch12_34=loss_deembedding_ch12_34,
    loss_deembedding_ch13_24=loss_deembedding_ch13_24,
    loss_deembedding_ch14_23=loss_deembedding_ch14_23,
    manhattan_text=manhattan_text,
    mmi1x2=mmi1x2,
    mmi2x2=mmi2x2,
    mzi=mzi,
    mzi_arm=mzi_arm,
    mzi_lattice=mzi_lattice,
    mzi_phase_shifter=mzi_phase_shifter,
    mzi_phase_shifter_90_90=mzi_phase_shifter_90_90,
    mzit=mzit,
    mzit_lattice=mzit_lattice,
    nxn=nxn,
    pad=pad,
    pad_array=pad_array,
    pads_shorted=pads_shorted,
    pcm_optical=pcm_optical,
    pixel=pixel,
    qrcode=qrcode,
    ramp=ramp,
    rectangle=rectangle,
    resistance_meander=resistance_meander,
    ring=ring,
    ring_double=ring_double,
    ring_single=ring_single,
    ring_single_array=ring_single_array,
    ring_single_dut=ring_single_dut,
    spiral=spiral,
    spiral_circular=spiral_circular,
    spiral_external_io=spiral_external_io,
    spiral_inner_io=spiral_inner_io,
    spiral_inner_io_fiber_single=spiral_inner_io_fiber_single,
    splitter_chain=splitter_chain,
    splitter_tree=splitter_tree,
    staircase=staircase,
    straight=straight,
    straight_array=straight_array,
    straight_heater_doped_rib=straight_heater_doped_rib,
    straight_heater_doped_strip=straight_heater_doped_strip,
    straight_heater_metal=straight_heater_metal,
    straight_heater_metal_90_90=straight_heater_metal_90_90,
    straight_heater_metal_undercut=straight_heater_metal_undercut,
    straight_heater_metal_undercut_90_90=straight_heater_metal_undercut_90_90,
    straight_heater_meander=straight_heater_meander,
    straight_pin=straight_pin,
    straight_pn=straight_pn,
    straight_pin_slot=straight_pin_slot,
    straight_rib=straight_rib,
    straight_rib_tapered=straight_rib_tapered,
    switch_tree=switch_tree,
    taper_cross_section_linear=taper_cross_section_linear,
    taper_cross_section_sine=taper_cross_section_sine,
    taper=taper,
    taper2=taper2,
    taper_0p5_to_3_l36=taper_0p5_to_3_l36,
    taper_from_csv=taper_from_csv,
    taper_strip_to_ridge=taper_strip_to_ridge,
    taper_strip_to_ridge_trenches=taper_strip_to_ridge_trenches,
    taper_w10_l100=taper_w10_l100,
    taper_w10_l150=taper_w10_l150,
    taper_w10_l200=taper_w10_l200,
    taper_w11_l200=taper_w11_l200,
    taper_w12_l200=taper_w12_l200,
    text=text,
    triangle=triangle,
    verniers=verniers,
    version_stamp=version_stamp,
    via=via,
    viac=viac,
    via1=via1,
    via2=via2,
    via_cutback=via_cutback,
    via_stack=via_stack,
    via_stack_slot=via_stack_slot,
    via_stack_heater=via_stack_heater,
    via_stack_slab=via_stack_slab,
    via_stack_with_offset=via_stack_with_offset,
    wire_corner=wire_corner,
    wire_sbend=wire_sbend,
    wire_straight=wire_straight,
)

_factory_passives = dict(
    bend_circular=bend_circular,
    bend_euler=bend_euler,
    bend_euler_s=bend_euler_s,
    bend_s=bend_s,
    cdc=cdc,
    coupler=coupler,
    coupler_adiabatic=coupler_adiabatic,
    coupler_asymmetric=coupler_asymmetric,
    coupler_full=coupler_full,
    coupler_ring=coupler_ring,
    coupler_symmetric=coupler_symmetric,
    crossing=crossing,
    crossing45=crossing45,
    taper_cross_section_linear=taper_cross_section_linear,
    taper_cross_section_sine=taper_cross_section_sine,
    taper=taper,
    taper2=taper2,
    taper_0p5_to_3_l36=taper_0p5_to_3_l36,
    taper_from_csv=taper_from_csv,
    taper_strip_to_ridge=taper_strip_to_ridge,
    taper_strip_to_ridge_trenches=taper_strip_to_ridge_trenches,
    taper_w10_l100=taper_w10_l100,
    taper_w10_l150=taper_w10_l150,
    taper_w10_l200=taper_w10_l200,
    taper_w11_l200=taper_w11_l200,
    taper_w12_l200=taper_w12_l200,
    mmi1x2=mmi1x2,
    mmi2x2=mmi2x2,
)
