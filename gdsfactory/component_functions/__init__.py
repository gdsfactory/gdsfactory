"""Component functions: undecorated builders behind ``gdsfactory.components``.

Each function here returns a new ``Component`` and is not cached. The cells in
``gdsfactory.components`` wrap them with ``@gf.cell``. A PDK can wrap them with
its own cell decorator instead.

Component functions look up sub-cells by name in the active PDK through
``get_component``, so PDK overrides apply inside composite components.
"""

from . import bends, mmis, mzis, tapers, waveguides
from ._get_component import ComponentFallbackWarning, get_component
from .bends import *
from .mmis import *
from .mzis import *
from .tapers import *
from .waveguides import *

__all__ = [
    "ComponentFallbackWarning",
    "bend_euler",
    "bend_euler_all_angle",
    "bend_euler_s",
    "bends",
    "crossing",
    "crossing45",
    "crossing_arm",
    "crossing_etched",
    "crossing_linear_taper",
    "get_component",
    "mmi1x2",
    "mmis",
    "mzi",
    "mzis",
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
    "taper",
    "taper_nc_sc",
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
