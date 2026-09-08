"""Core cross-section classes and type definitions.

You can define a path as list of points.
To create a component you need to extrude the path with a cross-section.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable
from typing import Any, Self, TypeAlias

import kfactory as kf
import numpy as np
from kfactory import (
    AsymmetricalCrossSection,
    DAsymmetricalCrossSection,
    DAsymmetricCrossSection,
    DCrossSection,
    SymmetricalCrossSection,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    field_serializer,
    model_validator,
)

from gdsfactory import typings
from gdsfactory.component import Component
from gdsfactory.config import CONF, ErrorType

nm = 1e-3

# Public gdsfactory cross-sections are the µm-based kfactory wrappers.  The
# DBU-based kfactory classes remain valid resolver inputs through CrossSectionSpec
# below, but are deliberately not aliased to ``gf.CrossSection``.
SymmetricCrossSection: TypeAlias = DCrossSection  # noqa: UP040
AsymmetricCrossSection: TypeAlias = DAsymmetricCrossSection  # noqa: UP040
CrossSection: TypeAlias = SymmetricCrossSection | AsymmetricCrossSection  # noqa: UP040


def validate_radius(
    cross_section: CrossSection,
    radius: float,
    error_type: ErrorType | None = None,
) -> None:
    """Validate an operation radius against native profile metadata."""
    radius_min = cross_section.radius_min or cross_section.radius
    if radius_min and radius < radius_min:
        message = f"min_bend_radius {radius} < cross_section.radius_min {radius_min}. "
        error_type = error_type or CONF.bend_radius_error_type
        if error_type == ErrorType.ERROR:
            raise ValueError(message)
        if error_type == ErrorType.WARNING:
            warnings.warn(message, stacklevel=3)


port_names_electrical: typings.IOPorts = ("e1", "e2")
port_types_electrical: typings.IOPorts = ("electrical", "electrical")
cladding_layers_optical: typings.Layers | None = None
cladding_offsets_optical: typings.Floats | None = None
cladding_simplify_optical: typings.Floats | None = None

deprecated = {
    "info",
    "add_pins_function_name",
    "add_pins_function_module",
    "min_length",
    "width_wide",
    "auto_widen",
    "auto_widen_minimum_length",
    "start_straight_length",
    "taper_length",
    "end_straight_length",
    "gap",
}

deprecated_pins = {
    "add_pins_function_name",
    "add_pins_function_module",
}

deprecated_routing = {
    "min_length",
    "width_wide",
    "auto_widen",
    "auto_widen_minimum_length",
    "start_straight_length",
    "taper_length",
    "end_straight_length",
    "gap",
}


class Section(BaseModel):
    """Section metadata used to extrude a path with a waveguide.

    Parameters:
        width: of the section (um). When `width_function` is set it takes \
                precedence during extrusion, so `width` acts as a nominal value.
        offset: center offset (um). When `offset_function` is set it takes \
                precedence during extrusion, so `offset` acts as a nominal value.
        insets: distance (um) in x to inset section relative to end of the Path \
                (i.e. (start inset, stop_inset)).
        layer: layer spec. If None does not draw the main section.
        port_names: Optional port names.
        port_types: optical, electrical, ...
        name: Optional Section name.
        hidden: hide layer.
        simplify: Optional Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting. \
                polygon by more than the value listed here will be removed.
        skip_transition: if True, this section is excluded from cross-section \
                transitions (will not be tapered between two CrossSections).
        width_function: parameterized function from 0 to 1.
        offset_function: parameterized function from 0 to 1.

         0

         │        ┌───────┐
                  │       │
         │        │ layer │
                  │◄─────►│
         │        │       │
                  │ width │
         │        └───────┘
                      |
         │
                      |
         ◄────────────►
            +offset
    """

    width: NonNegativeFloat = 0
    offset: float = 0
    insets: tuple[float, float] | None = None
    layer: typings.LayerSpec
    port_names: tuple[str | None, str | None] = (None, None)
    port_types: tuple[str, str] = ("optical", "optical")
    name: str | None = None
    hidden: bool = False
    simplify: float | None = None
    skip_transition: bool = False

    width_function: typings.WidthFunction | None = None
    offset_function: typings.OffsetFunction | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def generate_default_name(cls, data: Any) -> Any:
        if not data.get("name"):
            h = hashlib.md5(str(data).encode()).hexdigest()[:8]
            data["name"] = f"s_{h}"
        return data

    @model_validator(mode="after")
    def _require_width_value_or_function(self) -> Self:
        if self.width == 0 and self.width_function is None:
            raise ValueError("Section requires `width > 0` or a `width_function`.")
        return self

    @field_serializer("width_function")
    def serialize_width_function(
        self, func: typings.WidthFunction | None
    ) -> str | None:
        if func is None:
            return None
        t_values = np.linspace(0, 1, 11)
        return ",".join([str(round(width, 3)) for width in func(t_values)])

    @field_serializer("offset_function")
    def serialize_offset_function(
        self, func: typings.OffsetFunction | None
    ) -> str | None:
        if func is None:
            return None
        t_values = np.linspace(0, 1, 11)
        return ",".join([str(round(func(offset), 3)) for offset in t_values])


class ComponentAlongPath(BaseModel):
    """A ComponentAlongPath object to place along an extruded path.

    Parameters:
        component: to repeat along the path. The unrotated version should be oriented \
                for placement on a horizontal line.
        spacing: distance between component placements
        padding: minimum distance from the path start to the first component.
        y_offset: offset in y direction (um).
    """

    component: Component
    spacing: float
    padding: float = 0.0
    offset: float = 0.0

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


Sections = tuple[Section, ...]


class ExtrusionSection(BaseModel):
    """Extrusion metadata for one positional native cross-section strip.

    Native kfactory cross-sections describe geometry only.  Metadata that is
    consumed while drawing a path belongs here instead of on the profile.  The
    entries are positional: the first entry describes the main strip and the
    remaining entries describe the strips returned by ``get_sections()``.
    """

    port_names: tuple[str | None, str | None] = (None, None)
    port_types: tuple[str, str] = ("optical", "optical")
    hidden: bool = False
    simplify: float | None = None
    insets: tuple[float, float] | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExtrusionSpec(BaseModel):
    """Metadata used by path extrusion, independent of cross-section geometry."""

    sections: tuple[ExtrusionSection, ...] = Field(default_factory=tuple)
    components_along_path: tuple[ComponentAlongPath, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class SectionReference(BaseModel):
    """Positional reference to a native strip, identified by layer occurrence."""

    layer: typings.LayerSpec
    index: int = 0

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_index(self) -> Self:
        if self.index < 0:
            raise ValueError("SectionReference.index must be non-negative")
        return self


class TransitionSection(BaseModel):
    """One explicit start/end strip mapping for a native transition."""

    start: SectionReference | None = None
    end: SectionReference | None = None
    extrusion: ExtrusionSection = Field(default_factory=ExtrusionSection)

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _require_endpoint(self) -> Self:
        if self.start is None and self.end is None:
            raise ValueError("TransitionSection requires a start or end reference")
        return self


class SymmetricExtrusionSpec(BaseModel):
    """Explicit metadata/matching rules for symmetric transitions.

    An empty ``sections`` tuple asks the transition implementation to match
    strips by physical layer and occurrence.  Supplying entries makes the
    mapping explicit and also permits an endpoint to be ``None`` for an
    auxiliary strip that is intentionally ramped in or out.
    """

    sections: tuple[TransitionSection, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class AsymmetricExtrusionSpec(BaseModel):
    """Explicit strip mapping required for asymmetric transitions."""

    sections: tuple[TransitionSection, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class Transition(BaseModel, arbitrary_types_allowed=True):
    """Waveguide information to extrude a path between two native profiles.

    cladding_layers follow path shape

    Parameters:
        cross_section1: input cross_section.
        cross_section2: output cross_section.
        width_type: 'sine', 'linear', 'parabolic' or Callable. Sets the type of width \
                transition used if widths are different between the two input CrossSections.
        offset_type: 'sine', 'linear', 'parabolic' or Callable. Sets the type of offset \
                transition used if offsets are different between the two input CrossSections.
    """

    cross_section1: CrossSectionSpec
    cross_section2: CrossSectionSpec
    width_type: typings.WidthTypes | Callable[[float, float, float], float] = "sine"
    offset_type: typings.WidthTypes | Callable[[float, float, float], float] = "sine"
    extrusion_spec: SymmetricExtrusionSpec | AsymmetricExtrusionSpec | None = None
    core_width_profile: Callable[[Any], Any] | None = None

    @field_serializer("width_type")
    def serialize_width(
        self,
        width_type: typings.WidthTypes | Callable[[float, float, float], float],
    ) -> str:
        if isinstance(width_type, str):
            return width_type
        # TODO: implement callable serialization for width_type.
        raise NotImplementedError(
            "Serialization of callable width_type is not yet supported. "
            "Use a string value ('sine', 'linear', or 'parabolic') instead."
        )


class TransitionAsymmetric(BaseModel, arbitrary_types_allowed=True):
    """Waveguide information to extrude a path between two asymmetric profiles.

    Parameters:
        cross_section1: input cross_section.
        cross_section2: output cross_section.
        width_type1: transition type for lower edge width ('sine', 'linear', 'parabolic' or Callable).
        width_type2: transition type for upper edge width.
        offset_type1: transition type for lower edge offset.
        offset_type2: transition type for upper edge offset.
    """

    cross_section1: CrossSectionSpec
    cross_section2: CrossSectionSpec
    width_type1: typings.WidthTypes | Callable[[float, float, float], float] = "sine"
    width_type2: typings.WidthTypes | Callable[[float, float, float], float] = "sine"
    offset_type1: typings.WidthTypes | Callable[[float, float, float], float] = "sine"
    offset_type2: typings.WidthTypes | Callable[[float, float, float], float] = "sine"
    extrusion_spec: AsymmetricExtrusionSpec | None = None
    core_width_profile: Callable[[Any], Any] | None = None

    @field_serializer("width_type1")
    def serialize_width_type1(
        self,
        width_type1: typings.WidthTypes | Callable[[float, float, float], float],
    ) -> str:
        if isinstance(width_type1, str):
            return width_type1
        raise NotImplementedError(
            "Serialization of callable width_type1 is not yet supported. "
            "Use a string value ('sine', 'linear', or 'parabolic') instead."
        )

    @field_serializer("width_type2")
    def serialize_width_type2(
        self,
        width_type2: typings.WidthTypes | Callable[[float, float, float], float],
    ) -> str:
        if isinstance(width_type2, str):
            return width_type2
        raise NotImplementedError(
            "Serialization of callable width_type2 is not yet supported. "
            "Use a string value ('sine', 'linear', or 'parabolic') instead."
        )

    @field_serializer("offset_type1")
    def serialize_offset_type1(
        self,
        offset_type1: typings.WidthTypes | Callable[[float, float, float], float],
    ) -> str:
        if isinstance(offset_type1, str):
            return offset_type1
        raise NotImplementedError(
            "Serialization of callable offset_type1 is not yet supported. "
            "Use a string value ('sine', 'linear', or 'parabolic') instead."
        )

    @field_serializer("offset_type2")
    def serialize_offset_type2(
        self,
        offset_type2: typings.WidthTypes | Callable[[float, float, float], float],
    ) -> str:
        if isinstance(offset_type2, str):
            return offset_type2
        raise NotImplementedError(
            "Serialization of callable offset_type2 is not yet supported. "
            "Use a string value ('sine', 'linear', or 'parabolic') instead."
        )


type CrossSectionFactory = Callable[..., "CrossSection"]
type CrossSectionSpec = (
    str
    | dict[str, Any]
    | CrossSection
    | CrossSectionFactory
    | kf.CrossSection
    | SymmetricalCrossSection
    | kf.AsymmetricCrossSection
    | AsymmetricalCrossSection
    | DAsymmetricalCrossSection
)
