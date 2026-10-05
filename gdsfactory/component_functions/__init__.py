"""Component functions: undecorated builders behind ``gdsfactory.components``.

Each function here returns a new ``Component`` and is not cached. The cells in
``gdsfactory.components`` wrap them with ``@gf.cell``. A PDK can wrap them with
its own cell decorator instead.

Component functions look up sub-cells by name in the active PDK through
``get_component``, so PDK overrides apply inside composite components.
"""

from . import (
    bends,
    detectors,
    edge_couplers,
    mems,
    microfluidics,
    mmis,
    mzis,
    tapers,
    waveguides,
)
from ._get_component import ComponentFallbackWarning, get_component
from .bends import *
from .detectors import *
from .edge_couplers import *
from .mems import *
from .microfluidics import *
from .mmis import *
from .mzis import *
from .tapers import *
from .waveguides import *

__all__ = [
    "ComponentFallbackWarning",
    "anchored_flexure",
    "arrow_junction",
    "bend_circular",
    "bend_circular_all_angle",
    "bend_circular_heater",
    "bend_euler",
    "bend_euler_all_angle",
    "bend_euler_s",
    "bend_modified_hermite",
    "bend_modified_hermite_all_angle",
    "bend_modified_hermite_s",
    "bend_s",
    "bend_s_offset",
    "bend_topic",
    "bend_topic_all_angle",
    "bend_topic_s",
    "bends",
    "bent_beam",
    "bezier",
    "bolometer",
    "cantilever",
    "comb_drive",
    "crossing",
    "crossing45",
    "crossing_arm",
    "crossing_etched",
    "crossing_linear_taper",
    "detectors",
    "doubly_clamped_beam",
    "edge_coupler_array",
    "edge_coupler_array_with_loopback",
    "edge_coupler_silicon",
    "edge_couplers",
    "folded_spring",
    "ge_detector_straight_si_contacts",
    "gear",
    "get_component",
    "h_junction",
    "meander_channel",
    "mems",
    "microfluidics",
    "mmi",
    "mmi1x2",
    "mmi1x2_with_sbend",
    "mmi2x2",
    "mmi2x2_with_sbend",
    "mmi_90degree_hybrid",
    "mmi_tapered",
    "mmis",
    "mzi",
    "mzi_lattice",
    "mzi_lattice_mmi",
    "mzi_pads_center",
    "mzis",
    "mzit",
    "mzit_lattice",
    "ramp",
    "straight",
    "straight_all_angle",
    "straight_array",
    "straight_heater_doped_rib",
    "straight_heater_doped_strip",
    "straight_heater_meander",
    "straight_heater_meander_doped",
    "straight_heater_metal_simple",
    "straight_heater_metal_undercut",
    "straight_piecewise",
    "straight_pin",
    "straight_pin_slot",
    "t_junction",
    "taper",
    "taper_adiabatic",
    "taper_cross_section",
    "taper_from_csv",
    "taper_hecken",
    "taper_meander",
    "taper_nc_sc",
    "taper_parabolic",
    "taper_sc_nc",
    "taper_strip_to_ridge",
    "taper_strip_to_ridge_trenches",
    "tapers",
    "waveguides",
    "wire_corner",
    "wire_corner45",
    "wire_corner45_straight",
    "wire_corner_sections",
    "wire_straight",
]
