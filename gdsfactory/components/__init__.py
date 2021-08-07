""" NOTE: import order matters.
Only change the order if you know what you are doing

isort:skip_file
"""
import dataclasses
from gdsfactory.tech import LIBRARY

# level 0 components
from gdsfactory.components.array import array
from gdsfactory.components.array import array_2d
from gdsfactory.components.array_with_fanout import array_with_fanout
from gdsfactory.components.array_with_fanout import array_with_fanout_2d
from gdsfactory.components.array_with_via import array_with_via
from gdsfactory.components.array_with_via import array_with_via_2d
from gdsfactory.components.straight import straight
from gdsfactory.components.straight_heater import straight_heater
from gdsfactory.components.straight_heater import straight_with_heater
from gdsfactory.components.straight_pin import straight_pin
from gdsfactory.components.straight_array import straight_array

from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.bend_circular import bend_circular180
from gdsfactory.components.bend_circular_heater import bend_circular_heater
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.bezier import bezier
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.bend_euler import bend_euler180
from gdsfactory.components.bend_euler import bend_euler_s

from gdsfactory.components.coupler90 import coupler90
from gdsfactory.components.coupler90bend import coupler90bend
from gdsfactory.components.coupler_straight import coupler_straight
from gdsfactory.components.coupler_symmetric import coupler_symmetric
from gdsfactory.components.coupler_asymmetric import coupler_asymmetric
from gdsfactory.components.hline import hline

# basic shapes
from gdsfactory.components.circle import circle
from gdsfactory.components.compass import compass
from gdsfactory.components.cross import cross
from gdsfactory.components.crossing_waveguide import crossing
from gdsfactory.components.crossing_waveguide import crossing45
from gdsfactory.components.crossing_waveguide import compensation_path
from gdsfactory.components.ellipse import ellipse
from gdsfactory.components.rectangle import rectangle
from gdsfactory.components.ring import ring
from gdsfactory.components.extension import extend_ports
from gdsfactory.components.taper import taper
from gdsfactory.components.taper import taper_strip_to_ridge
from gdsfactory.components.taper_from_csv import taper_0p5_to_3_l36
from gdsfactory.components.text import text
from gdsfactory.components.L import L
from gdsfactory.components.C import C
from gdsfactory.components.bbox import bbox
from gdsfactory.components.nxn import nxn
from gdsfactory.components.ramp import ramp
from gdsfactory.components.die import die
from gdsfactory.components.die_bbox import die_bbox

# optical test structures
from gdsfactory.components.version_stamp import version_stamp
from gdsfactory.components.version_stamp import qrcode
from gdsfactory.components.manhattan_font import manhattan_text
from gdsfactory.components.logo import logo
from gdsfactory.components.align import align_wafer
from gdsfactory.components.cutback_bend import cutback_bend90
from gdsfactory.components.cutback_bend import cutback_bend180
from gdsfactory.components.cutback_component import cutback_component
from gdsfactory.components.cutback_component import cutback_component_flipped

from gdsfactory.components.pcm.litho_calipers import litho_calipers
from gdsfactory.components.pcm.litho_steps import litho_steps
from gdsfactory.components.pcm.verniers import verniers
from gdsfactory.components.litho_ruler import litho_ruler


from gdsfactory.components.grating_coupler.elliptical import (
    grating_coupler_elliptical_te,
)
from gdsfactory.components.grating_coupler.elliptical import (
    grating_coupler_elliptical_tm,
)
from gdsfactory.components.grating_coupler.elliptical2 import (
    grating_coupler_elliptical2,
)
from gdsfactory.components.grating_coupler.uniform import grating_coupler_uniform
from gdsfactory.components.grating_coupler.uniform_optimized import (
    grating_coupler_uniform_optimized,
)
from gdsfactory.components.grating_coupler.grating_coupler_tree import (
    grating_coupler_tree,
)
from gdsfactory.components.grating_coupler.elliptical_trenches import grating_coupler_te
from gdsfactory.components.grating_coupler.elliptical_trenches import grating_coupler_tm
from gdsfactory.components.grating_coupler.grating_coupler_loss import (
    grating_coupler_loss,
)
from gdsfactory.components.delay_snake import delay_snake
from gdsfactory.components.delay_snake2 import delay_snake2
from gdsfactory.components.delay_snake3 import delay_snake3
from gdsfactory.components.spiral import spiral
from gdsfactory.components.spiral_inner_io import spiral_inner_io_euler
from gdsfactory.components.spiral_inner_io import spiral_inner_io
from gdsfactory.components.spiral_inner_io import spiral_inner_io_with_gratings
from gdsfactory.components.spiral_external_io import spiral_external_io
from gdsfactory.components.spiral_circular import spiral_circular
from gdsfactory.components.cdc import cdc
from gdsfactory.components.dbr import dbr
from gdsfactory.components.dbr2 import dbr2

# electrical
from gdsfactory.components.wire import wire_corner
from gdsfactory.components.wire import wire_straight
from gdsfactory.components.wire_sbend import wire_sbend
from gdsfactory.components.electrical.pad import pad
from gdsfactory.components.electrical.pad import pad_array
from gdsfactory.components.electrical.pad import pad_array_2d
from gdsfactory.components.via import via
from gdsfactory.components.via import via1
from gdsfactory.components.via import via2
from gdsfactory.components.via import via3
from gdsfactory.components.via_stack import via_stack
from gdsfactory.components.via_stack_with_offset import via_stack_with_offset
from gdsfactory.components.electrical.pads_shorted import pads_shorted

# electrical PCM
from gdsfactory.components.resistance_meander import resistance_meander
from gdsfactory.components.via_cutback import via_cutback

# level 1 components
from gdsfactory.components.cavity import cavity
from gdsfactory.components.coupler import coupler
from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.coupler_adiabatic import coupler_adiabatic
from gdsfactory.components.coupler_full import coupler_full
from gdsfactory.components.disk import disk
from gdsfactory.components.ring_single import ring_single
from gdsfactory.components.ring_single_array import ring_single_array
from gdsfactory.components.ring_double import ring_double
from gdsfactory.components.mmi1x2 import mmi1x2
from gdsfactory.components.mmi2x2 import mmi2x2
from gdsfactory.components.mzi import mzi
from gdsfactory.components.mzi_phase_shifter import mzi_phase_shifter
from gdsfactory.components.mzit import mzit
from gdsfactory.components.mzi_lattice import mzi_lattice
from gdsfactory.components.mzit_lattice import mzit_lattice
from gdsfactory.components.loop_mirror import loop_mirror
from gdsfactory.components.fiber import fiber
from gdsfactory.components.fiber_array import fiber_array

# level 2 components
from gdsfactory.components.awg import awg
from gdsfactory.components.component_lattice import component_lattice
from gdsfactory.components.component_sequence import component_sequence
from gdsfactory.components.splitter_tree import splitter_tree
from gdsfactory.components.splitter_chain import splitter_chain


LIBRARY.register(
    [
        array,
        array_2d,
        array_with_fanout,
        array_with_fanout_2d,
        array_with_via,
        array_with_via_2d,
        C,
        L,
        align_wafer,
        awg,
        bbox,
        bend_circular180,
        bend_circular,
        bend_circular_heater,
        bend_euler180,
        bend_euler,
        bend_euler_s,
        bend_s,
        bezier,
        cavity,
        cdc,
        circle,
        compass,
        compensation_path,
        component_lattice,
        component_sequence,
        coupler90,
        coupler90bend,
        coupler,
        coupler_adiabatic,
        coupler_asymmetric,
        coupler_full,
        coupler_ring,
        coupler_straight,
        coupler_symmetric,
        cross,
        crossing45,
        crossing,
        cutback_bend180,
        cutback_bend90,
        cutback_component,
        cutback_component_flipped,
        dbr2,
        dbr,
        delay_snake,
        delay_snake2,
        delay_snake3,
        disk,
        die,
        die_bbox,
        ellipse,
        fiber,
        fiber_array,
        grating_coupler_elliptical2,
        grating_coupler_elliptical_te,
        grating_coupler_elliptical_tm,
        grating_coupler_loss,
        grating_coupler_te,
        grating_coupler_tm,
        grating_coupler_tree,
        grating_coupler_uniform,
        grating_coupler_uniform_optimized,
        hline,
        litho_calipers,
        litho_steps,
        litho_ruler,
        logo,
        loop_mirror,
        manhattan_text,
        mmi1x2,
        mmi2x2,
        mzi,
        mzi_phase_shifter,
        mzi_lattice,
        mzit,
        mzit_lattice,
        nxn,
        pad,
        pad_array,
        pad_array_2d,
        pads_shorted,
        qrcode,
        ramp,
        rectangle,
        ring,
        ring_double,
        ring_single,
        ring_single_array,
        spiral,
        spiral_circular,
        spiral_external_io,
        spiral_inner_io,
        spiral_inner_io_euler,
        spiral_inner_io_with_gratings,
        splitter_chain,
        splitter_tree,
        taper,
        taper_0p5_to_3_l36,
        taper_strip_to_ridge,
        extend_ports,
        resistance_meander,
        via_cutback,
        text,
        via_stack,
        via_stack_with_offset,
        verniers,
        version_stamp,
        via1,
        via2,
        via3,
        via,
        straight,
        straight_array,
        straight_heater,
        straight_pin,
        straight_with_heater,
        wire_straight,
        wire_corner,
        wire_sbend,
    ]
)


def library(component_type: str, **kwargs):
    """Returns a component with settings.
    from TECH.component_settings.component_type

    Args:
        component_type: library
        **kwargs: component_settings

    """
    from gdsfactory.tech import TECH
    import gdsfactory as gf

    settings = getattr(TECH.component_settings, component_type)
    settings = dataclasses.asdict(settings) if settings else {}
    component_type = settings.pop("component_type", component_type)
    settings.update(**kwargs)

    if isinstance(component_type, gf.Component):
        return component_type
    elif callable(component_type):
        return component_type(**settings)
    elif component_type not in LIBRARY.library.keys():
        raise ValueError(
            f"component_type = {component_type} not in: \n"
            + "\n".join(LIBRARY.library.keys())
        )
    return LIBRARY.library[component_type](**settings)


component_names_skip_test = [
    "text",
    "component_sequence",
    "compensation_path",
    "component_lattice",
    "version_stamp",
    "resistance_meander",
]
component_names_skip_test_ports = ["coupler"]

container_names = ["cavity", "ring_single_dut"]
component_names = (
    set(LIBRARY.factory.keys()) - set(component_names_skip_test) - set(container_names)
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

__all__ = list(LIBRARY.factory.keys()) + container_names + ["extend_ports_list"]
component_factory = LIBRARY.factory

if __name__ == "__main__":
    for component_name in component_names:
        try:
            ci = LIBRARY.factory[component_name]()
        except Exception:
            print(f"error building {component_name}")
            raise Exception
    ci.show()
