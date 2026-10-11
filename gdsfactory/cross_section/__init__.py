"""Cross-section package for gdsfactory.

You can define a path as list of points.
To create a component you need to extrude the path with a cross-section.

Factories return kfactory DCrossSection or DAsymmetricCrossSection objects.
Extrusion settings belong to path.extrude(), not to the transverse profile.
"""

# Re-export everything from submodules for backward compatibility
# fmt: off

# --- base classes, type aliases, and constants ---
from gdsfactory.cross_section.base import (
    AsymmetricCrossSection,
    CrossSection,
    CrossSectionFactory,
    CrossSectionSpec,
    Sections,
    SectionSpec,
    SymmetricCrossSection,
    Transition,
    TransitionAsymmetric,
    cladding_layers_optical,
    cladding_offsets_optical,
    nm,
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
    strip_rib_tip,
)

# --- utilities: factory function, decorator, registry, introspection ---
from gdsfactory.cross_section.utils import (
    CrossSectionCallable,
    P,
    cross_section,
    cross_sections,
    get_cross_sections,
    is_cross_section,
    section_cross_section,
    validate_radius,
    with_width,
    xsection,
)

# fmt: on

__all__ = [
    "AsymmetricCrossSection",
    "CrossSection",
    "CrossSectionCallable",
    "CrossSectionFactory",
    "CrossSectionSpec",
    "P",
    "SectionSpec",
    "Sections",
    "SymmetricCrossSection",
    "Transition",
    "TransitionAsymmetric",
    "cladding_layers_optical",
    "cladding_offsets_optical",
    "cross_section",
    "cross_sections",
    "get_cross_sections",
    "gs",
    "gsg",
    "heater_metal",
    "is_cross_section",
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
    "radius_nitride",
    "radius_rib",
    "rib",
    "rib2",
    "rib_bbox",
    "rib_heater_doped",
    "rib_heater_doped_via_stack",
    "rib_with_trenches",
    "section_cross_section",
    "slot",
    "strip",
    "strip_heater_doped",
    "strip_heater_metal",
    "strip_heater_metal_undercut",
    "strip_nitride_tip",
    "strip_rib_tip",
    "validate_radius",
    "with_width",
    "xsection",
]

cross_sections["metal_routing"] = metal3
