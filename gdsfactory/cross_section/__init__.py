"""Cross-section package for gdsfactory.

You can define a path as list of points.
To create a component you need to extrude the path with a cross-section.

This package provides the core CrossSection class, preset cross-section
factories, P-N junction definitions, heater variants, and utility functions.

All public names are re-exported here for full backward compatibility with
``from gdsfactory.cross_section import ...`` and
``from gdsfactory import cross_section``.
"""

# Re-export everything from submodules for backward compatibility
# fmt: off

# --- base classes, type aliases, and constants ---
from gdsfactory.cross_section.base import (
    ComponentAlongPath,
    CrossSection,
    CrossSectionFactory,
    CrossSectionSpec,
    Section,
    Sections,
    Transition,
    TransitionAsymmetric,
    cladding_layers_optical,
    cladding_offsets_optical,
    cladding_simplify_optical,
    deprecated,
    deprecated_pins,
    deprecated_routing,
    nm,
    port_names_electrical,
    port_types_electrical,
)

# --- heater cross-section factories ---
from gdsfactory.cross_section.heater import (
    rib_heater_doped,
    rib_heater_doped_via_stack,
    strip_heater_doped,
    strip_heater_metal,
    strip_heater_metal_undercut,
)

# --- P-N junction cross-section factories ---
from gdsfactory.cross_section.pn_junction import (
    l_wg_doped_with_trenches,
    pin,
    pn,
    pn_ge_detector_si_contacts,
    pn_with_trenches,
    pn_with_trenches_asymmetric,
)

# --- preset cross-section factories ---
from gdsfactory.cross_section.presets import (
    gs,
    gsg,
    heater_metal,
    l_with_trenches,
    metal1,
    metal2,
    metal3,
    metal_routing,
    nitride,
    npp,
    radius_nitride,
    radius_rib,
    rib,
    rib2,
    rib_bbox,
    rib_with_trenches,
    slot,
    strip,
    strip_nitride_tip,
    strip_no_ports,
    strip_rib_tip,
)

# --- utilities: factory function, decorator, registry, introspection ---
from gdsfactory.cross_section.utils import (
    CrossSectionCallable,
    P,
    _cross_section_default_names,
    cross_section,
    cross_sections,
    get_cross_sections,
    is_cross_section,
    xsection,
)

# fmt: on

__all__ = [
    # base
    "ComponentAlongPath",
    "CrossSection",
    # utils
    "CrossSectionCallable",
    "CrossSectionFactory",
    "CrossSectionSpec",
    "P",
    "Section",
    "Sections",
    "Transition",
    "TransitionAsymmetric",
    "_cross_section_default_names",
    "cladding_layers_optical",
    "cladding_offsets_optical",
    "cladding_simplify_optical",
    "cross_section",
    "cross_sections",
    "deprecated",
    "deprecated_pins",
    "deprecated_routing",
    "get_cross_sections",
    # presets
    "gs",
    "gsg",
    "heater_metal",
    "is_cross_section",
    # pn_junction
    "l_wg_doped_with_trenches",
    "l_with_trenches",
    "metal1",
    "metal2",
    "metal3",
    "metal_routing",
    "nitride",
    "nm",
    "npp",
    "pin",
    "pn",
    "pn_ge_detector_si_contacts",
    "pn_with_trenches",
    "pn_with_trenches_asymmetric",
    "port_names_electrical",
    "port_types_electrical",
    "radius_nitride",
    "radius_rib",
    "rib",
    "rib2",
    "rib_bbox",
    # heater
    "rib_heater_doped",
    "rib_heater_doped_via_stack",
    "rib_with_trenches",
    "slot",
    "strip",
    "strip_heater_doped",
    "strip_heater_metal",
    "strip_heater_metal_undercut",
    "strip_nitride_tip",
    "strip_no_ports",
    "strip_rib_tip",
    "xsection",
]
