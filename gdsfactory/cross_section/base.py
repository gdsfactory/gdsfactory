"""Core cross-section classes and type definitions.

You can define a path as list of points.
To create a component you need to extrude the path with a cross-section.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable
from typing import Any, Self, TypeAlias

import numpy as np
from kfactory import DCrossSection, SymmetricalCrossSection, logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PrivateAttr,
    field_serializer,
    model_validator,
)

from gdsfactory import typings
from gdsfactory.component import Component
from gdsfactory.config import CONF, ErrorType

nm = 1e-3


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
    """CrossSection to extrude a path with a waveguide.

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
        width_function: parameterized function from 0 to 1.
        offset_function: parameterized function from 0 to 1.

    .. code::

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


class CrossSection(BaseModel):
    """Waveguide information to extrude a path.

    Parameters:
        sections: tuple of Sections(width, offset, layer, ports).
        components_along_path: tuple of ComponentAlongPaths.
        radius: default bend radius for routing (um).
        radius_min: minimum acceptable bend radius.
        bbox_layers: layer to add as bounding box.
        bbox_offsets: offset to add to the bounding box.

    .. code::


           ┌────────────────────────────────────────────────────────────┐
           │                                                            │
           │                                                            │
           │                   boox_layer                               │
           │                                                            │
           │         ┌──────────────────────────────────────┐           │
           │         │                            ▲         │bbox_offset│
           │         │                            │         ├──────────►│
           │         │           cladding_offset  │         │           │
           │         │                            │         │           │
           │         ├─────────────────────────▲──┴─────────┤           │
           │         │                         │            │           │
        ─ ─┤         │           core   width  │            │           ├─ ─ center
           │         │                         │            │           │
           │         ├─────────────────────────▼────────────┤           │
           │         │                                      │           │
           │         │                                      │           │
           │         │                                      │           │
           │         │                                      │           │
           │         └──────────────────────────────────────┘           │
           │                                                            │
           │                                                            │
           │                                                            │
           └────────────────────────────────────────────────────────────┘

    """

    sections: Sections = Field(default_factory=tuple)
    components_along_path: tuple[ComponentAlongPath, ...] = Field(default_factory=tuple)
    radius: float | None = None
    radius_min: float | None = None
    bbox_layers: typings.LayerSpecs | None = None
    bbox_offsets: typings.Floats | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)
    _name: str = PrivateAttr("")
    _dcross_section: DCrossSection | None = PrivateAttr()

    def validate_radius(
        self, radius: float, error_type: ErrorType | None = None
    ) -> None:
        radius_min = self.radius_min or self.radius

        if radius_min and radius < radius_min:
            message = (
                f"min_bend_radius {radius} < CrossSection.radius_min {radius_min}. "
            )

            error_type = error_type or CONF.bend_radius_error_type

            if error_type == ErrorType.ERROR:
                raise ValueError(message)

            if error_type == ErrorType.WARNING:
                warnings.warn(message, stacklevel=3)

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        h = hashlib.md5(str(self).encode()).hexdigest()[:8]
        return f"xs_{h}"

    @property
    def width(self) -> float:
        return self.sections[0].width

    @property
    def layer(self) -> typings.LayerSpec:
        return self.sections[0].layer

    def append_sections(self, sections: Sections) -> Self:
        """Append sections to the cross_section."""
        new_sections = list(self.sections) + list(sections)
        return self.model_copy(update={"sections": tuple(new_sections)})

    def __getitem__(self, key: str) -> Section:
        """Returns the section with the given name."""
        key_to_section = {s.name: s for s in self.sections}
        if key in key_to_section:
            return key_to_section[key]
        raise KeyError(f"{key} not in {list(key_to_section.keys())}")

    @property
    def hash(self) -> str:
        """Returns a hash of the cross_section."""
        return hashlib.md5(str(self).encode()).hexdigest()

    def copy(
        self,
        width: float | None = None,
        layer: typings.LayerSpec | None = None,
        width_function: typings.WidthFunction | None = None,
        offset_function: typings.OffsetFunction | None = None,
        sections: Sections | None = None,
        **kwargs: Any,
    ) -> CrossSection:
        """Returns copy of the cross_section with new parameters.

        Args:
            width: of the section (um). Defaults to current width.
            layer: layer spec. Defaults to current layer.
            width_function: parameterized function from 0 to 1.
            offset_function: parameterized function from 0 to 1.
            sections: a tuple of Sections, to replace the original sections
            kwargs: additional parameters to update.

        Keyword Args:
            sections: tuple of Sections(width, offset, layer, ports).
            components_along_path: tuple of ComponentAlongPaths.
            radius: route bend radius (um).
            bbox_layers: layer to add as bounding box.
            bbox_offsets: offset to add to the bounding box.
            _name: name of the cross_section.

        """
        for kwarg in kwargs:
            if kwarg not in dict(self):
                raise ValueError(f"{kwarg!r} not in CrossSection")

        xs_original = self

        if width_function or offset_function or width or layer or sections:
            if sections is None:
                section_list = list(self.sections)
            else:
                section_list = list(sections)

            section_list = [s.model_copy() for s in section_list]
            section_list[0] = section_list[0].model_copy(
                update={
                    "width_function": width_function,
                    "offset_function": offset_function,
                    "width": width or self.width,
                    "layer": layer or self.layer,
                }
            )
            xs = self.model_copy(update={"sections": tuple(section_list), **kwargs})
            if xs != xs_original:
                xs._name = f"xs_{xs.hash}"
            return xs

        xs = self.model_copy(update=kwargs)
        if xs != xs_original:
            xs._name = f"xs_{xs.hash}"
        return xs

    def mirror(self) -> CrossSection:
        """Returns a mirrored copy of the cross_section."""
        sections = [s.model_copy(update=dict(offset=-s.offset)) for s in self.sections]
        return self.model_copy(update={"sections": tuple(sections)})

    def add_bbox(
        self,
        component: typings.AnyComponentT,
        top: float | None = None,
        bottom: float | None = None,
        right: float | None = None,
        left: float | None = None,
    ) -> typings.AnyComponentT:
        """Add bounding box layers to a component.

        Args:
            component: to add layers.
            top: top padding.
            bottom: bottom padding.
            right: right padding.
            left: left padding.
        """
        from gdsfactory.add_padding import get_padding_points

        c = component
        if self.bbox_layers and self.bbox_offsets:
            padding: list[list[typings.Coordinate]] = []
            for offset in self.bbox_offsets:
                points = get_padding_points(
                    component=c,
                    default=0,
                    top=top if top is not None else offset,
                    bottom=bottom if bottom is not None else offset,
                    right=right if right is not None else offset,
                    left=left if left is not None else offset,
                )
                padding.append(points)

            for layer, points in zip(self.bbox_layers, padding, strict=False):
                c.add_polygon(points, layer=layer)
        return c

    def get_xmin_xmax(self) -> tuple[float, float]:
        """Returns the min and max extent of the cross_section across all sections."""
        main_width = self.width
        main_offset = self.sections[0].offset
        xmin = main_offset - main_width / 2
        xmax = main_offset + main_width / 2
        for section in self.sections:
            width = section.width
            offset = section.offset
            xmin = min(xmin, offset - width / 2)
            xmax = max(xmax, offset + width / 2)

        return xmin, xmax


CrossSection.model_rebuild()


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


CrossSectionFactory: TypeAlias = Callable[..., "CrossSection"]
CrossSectionSpec: TypeAlias = (
    CrossSection
    | str
    | dict[str, Any]
    | CrossSectionFactory
    | SymmetricalCrossSection
    | DCrossSection
)
