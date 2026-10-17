"""Cross-section types and extrusion-time transitions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import kfactory as kf
from kfactory import DAsymmetricCrossSection as AsymmetricCrossSection
from kfactory import DCrossSection as SymmetricCrossSection
from kfactory import DCrossSectionLayer
from pydantic import BaseModel, field_serializer

from gdsfactory import typings


class Section(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """A strip of a cross section together with its extrusion settings.

    Use it in ``cross_section(sections=...)`` instead of a ``(layer, min, max)``
    tuple to give a strip its own ports, insets, visibility or simplification.
    The settings are stored once per cross-section geometry and apply to every
    extrusion of it, unless ``extrude()`` overrides them.

    Attributes:
        layer: strip layer.
        section_min: lower edge of the strip (um).
        section_max: upper edge of the strip (um).
        port_names: (start, end) port names. A None name creates no port.
        port_types: port type, or (start, end) port types, which must match.
        insets: (start, end) distances by which the strip is shortened at the
            path ends (um).
        hidden: whether the strip is left out of the extrusion.
        simplify: polygon simplification tolerance of the strip (um).
    """

    layer: typings.LayerSpec | kf.kdb.LayerInfo
    section_min: float
    section_max: float
    port_names: tuple[str | None, str | None] | None = None
    port_types: str | tuple[str, str] | None = None
    insets: tuple[float, float] | None = None
    hidden: bool = False
    simplify: float | None = None

    def __init__(
        self,
        layer: typings.LayerSpec | kf.kdb.LayerInfo,
        section_min: float,
        section_max: float,
        **kwargs: Any,
    ) -> None:
        """Create a strip from its layer and bounds, plus extrusion settings."""
        super().__init__(
            layer=layer, section_min=section_min, section_max=section_max, **kwargs
        )

    @property
    def port_type(self) -> str | None:
        """The single port type of both ends."""
        if not isinstance(self.port_types, tuple):
            return self.port_types
        start, end = self.port_types
        if start != end:
            raise ValueError(
                f"Both ports of a strip must have the same type, got {self.port_types}"
            )
        return start

    def extrude_settings(self) -> tuple[Any, ...]:
        """The extrusion settings, without the geometry."""
        return (
            self.port_names,
            self.port_type,
            self.insets,
            self.hidden,
            self.simplify,
        )

    def has_extrude_settings(self) -> bool:
        """Whether any extrusion setting differs from the default."""
        return self.extrude_settings() != (None, None, None, False, None)


# Keep the runtime union usable with isinstance(), unlike a TypeAliasType.
CrossSection: TypeAlias = SymmetricCrossSection | AsymmetricCrossSection  # noqa: UP040
type SectionSpec = (
    Section
    | DCrossSectionLayer
    | tuple[typings.LayerSpec | kf.kdb.LayerInfo, float, float]
)
type Sections = Sequence[SectionSpec]
type CrossSectionFactory = Callable[..., CrossSection]
type CrossSectionSpec = (
    CrossSection
    | kf.SymmetricalCrossSection
    | kf.AsymmetricalCrossSection
    | str
    | dict[str, Any]
    | CrossSectionFactory
)
nm = 1e-3
cladding_layers_optical = None
cladding_offsets_optical = None
port_names_electrical: typings.IOPorts = ("e1", "e2")
port_types_electrical: typings.IOPorts = ("electrical", "electrical")


class Transition(BaseModel, arbitrary_types_allowed=True):
    """Waveguide information to extrude a path between two CrossSection.

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
    """Waveguide information to extrude a path between two CrossSection with asymmetric transitions.

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
