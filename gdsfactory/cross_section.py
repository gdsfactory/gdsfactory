"""You can define a path as list of points.

To create a component you need to extrude the path with a cross-section.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Sequence
from functools import partial, wraps
from inspect import getmembers, isbuiltin, isfunction
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any, ParamSpec, Protocol, Self, TypeAlias

import numpy as np
from kfactory import DCrossSection, SymmetricalCrossSection, logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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
        width: of the section (um) or parameterized function from 0 to 1. \
                the width at t==0 is the width at the beginning of the Path. \
                the width at t==1 is the width at the end.
        offset: center offset (um) or function parameterized function from 0 to 1. \
                the offset at t==0 is the offset at the beginning of the Path. \
                the offset at t==1 is the offset at the end.
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

    width: float
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

            elif error_type == ErrorType.WARNING:
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
        else:
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

    # def apply_enclosure(self, component: Component) -> None:
    #     """Apply enclosure to a target component according to :class:`CrossSection`."""

    #     enclosure = kf.LayerEnclosure(
    #         dsections=[(layer_tuple, layer_offset) for zip(self.bbox_layers, self.bbox_offsets)],
    #         main_layer=LAYER.SLAB90,
    #         name="enclosures",
    #         kcl=kf.kcl,
    #     )
    #     kf.kcl.layer_enclosures = kf.kcell.LayerEnclosureModel(
    #         enclosure_map=dict(enclosure_rc=enclosure_rc)
    #     )

    #     kf.kcl.enclosure = kf.DKCellEnclosure(
    #         enclosures=[enclosure_rc],
    #     )
    #     component.kcl.enclosure.apply_minkowski_y(component)

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

            for layer, points in zip(self.bbox_layers, padding):
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
        raise NotImplementedError("TODO")
        t_values = np.linspace(0, 1, 10)
        return ",".join(
            [str(round(width, 3)) for width in width_type(t_values, *self.width)]
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
cross_sections: dict[str, CrossSectionFactory] = {}
_cross_section_default_names: dict[str, str] = {}

P = ParamSpec("P")


class CrossSectionCallable(Protocol[P]):
    __name__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> CrossSection: ...


def xsection(func: CrossSectionCallable[P]) -> CrossSectionCallable[P]:
    """Decorator to register a cross section function.

    Ensures that the cross-section name matches the name of the function that generated it when created using default parameters

    .. code-block:: python

        @xsection
        def xs_sc(width=TECH.width_sc, radius=TECH.radius_sc):
            return gf.cross_section.cross_section(width=width, radius=radius)
    """
    default_xs = func()  # type: ignore[call-arg]
    _cross_section_default_names[default_xs.name] = func.__name__

    @wraps(func)
    def newfunc(*args: P.args, **kwargs: P.kwargs) -> CrossSection:
        xs = func(*args, **kwargs)
        if xs.name in _cross_section_default_names:
            xs._name = _cross_section_default_names[xs.name]
        return xs

    cross_sections[func.__name__] = newfunc
    return newfunc


def cross_section(
    width: float = 0.5,
    offset: float = 0,
    layer: typings.LayerSpec = "WG",
    sections: Sections | None = None,
    port_names: typings.IOPorts = ("o1", "o2"),
    port_types: typings.IOPorts = ("optical", "optical"),
    bbox_layers: typings.LayerSpecs | None = None,
    bbox_offsets: typings.Floats | None = None,
    cladding_layers: typings.LayerSpecs | None = None,
    cladding_offsets: typings.Floats | None = None,
    cladding_simplify: typings.Floats | None = None,
    cladding_centers: typings.Floats | None = None,
    radius: float | None = 10.0,
    radius_min: float | None = None,
    main_section_name: str = "_default",
) -> CrossSection:
    """Return CrossSection.

    Args:
        width: main Section width (um).
        offset: main Section center offset (um).
        layer: main section layer.
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        bbox_layers: list of layers bounding boxes to extrude.
        bbox_offsets: list of offset from bounding box edge.
        cladding_layers: list of layers to extrude.
        cladding_offsets: list of offset from main Section edge.
        cladding_simplify: Optional Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting. \
                polygon by more than the value listed here will be removed.
        cladding_centers: center offset for each cladding layer. Defaults to 0.
        radius: routing bend radius (um).
        radius_min: min acceptable bend radius.
        main_section_name: name of the main section. Defaults to _default

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.cross_section(width=0.5, offset=0, layer='WG')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()

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
    section_list: list[Section] = list(sections or [])
    cladding_simplify_not_none: list[float | None] | None = None
    cladding_offsets_not_none: list[float] | None = None
    cladding_centers_not_none: list[float] | None = None
    if cladding_layers:
        cladding_simplify_not_none = list(
            cladding_simplify or (None,) * len(cladding_layers)
        )
        cladding_offsets_not_none = list(
            cladding_offsets or (0,) * len(cladding_layers)
        )
        cladding_centers_not_none = list(cladding_centers or [0] * len(cladding_layers))

        if (
            len(
                {
                    len(x)
                    for x in (
                        cladding_layers,
                        cladding_offsets_not_none,
                        cladding_simplify_not_none,
                        cladding_centers_not_none,
                    )
                }
            )
            > 1
        ):
            raise ValueError(
                f"{len(cladding_layers)=}, "
                f"{len(cladding_offsets_not_none)=}, "
                f"{len(cladding_simplify_not_none)=}, "
                f"{len(cladding_centers_not_none)=} must have same length"
            )
    s = [
        Section(
            width=width,
            offset=offset,
            layer=layer,
            port_names=port_names,
            port_types=port_types,
            name=main_section_name,
        )
    ] + section_list

    if (
        cladding_layers
        and cladding_offsets_not_none
        and cladding_simplify_not_none
        and cladding_centers_not_none
    ):
        s += [
            Section(
                width=width + 2 * offset,
                layer=layer,
                simplify=simplify,
                offset=center,
                name=f"cladding_{i}",
            )
            for i, (layer, offset, simplify, center) in enumerate(
                zip(
                    cladding_layers,
                    cladding_offsets_not_none,
                    cladding_simplify_not_none,
                    cladding_centers_not_none,
                )
            )
        ]
    return CrossSection(
        sections=tuple(s),
        radius=radius,
        radius_min=radius_min,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
    )


radius_nitride = 20
radius_rib = 20


@xsection
def strip(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    radius: float = 10.0,
    radius_min: float = 5,
    **kwargs: Any,
) -> CrossSection:
    """Return Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        **kwargs,
    )


@xsection
def strip_no_ports(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    radius: float = 10.0,
    radius_min: float = 5,
    port_names: typings.IOPorts = ("", ""),
    **kwargs: Any,
) -> CrossSection:
    """Return Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        port_names=port_names,
        **kwargs,
    )


@xsection
def rib(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    radius: float = radius_rib,
    radius_min: float | None = None,
    cladding_layers: typings.LayerSpecs = ("SLAB90",),
    cladding_offsets: typings.Floats = (3,),
    cladding_simplify: typings.Floats = (50 * nm,),
    **kwargs: Any,
) -> CrossSection:
    """Return Rib cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        cladding_layers=cladding_layers,
        cladding_offsets=cladding_offsets,
        cladding_simplify=cladding_simplify,
        **kwargs,
    )


@xsection
def rib_bbox(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    radius: float = radius_rib,
    radius_min: float | None = None,
    bbox_layers: typings.LayerSpecs = ("SLAB90",),
    bbox_offsets: typings.Floats = (3,),
    **kwargs: Any,
) -> CrossSection:
    """Return Rib cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        **kwargs,
    )


@xsection
def rib2(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "SLAB90",
    radius: float = radius_rib,
    radius_min: float | None = None,
    width_slab: float = 6,
    **kwargs: Any,
) -> CrossSection:
    """Return Rib cross_section."""
    sections = (
        Section(width=width_slab, layer=layer_slab, name="slab", simplify=50 * nm),
    )
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        sections=sections,
        **kwargs,
    )


@xsection
def nitride(
    width: float = 1.0,
    layer: typings.LayerSpec = "WGN",
    radius: float = radius_nitride,
    radius_min: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        **kwargs,
    )


@xsection
def strip_rib_tip(
    width: float = 0.5,
    width_tip: float = 0.2,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "SLAB90",
    radius: float = 10.0,
    radius_min: float | None = 5,
    **kwargs: Any,
) -> CrossSection:
    """Return Strip cross_section."""
    sections = (Section(width=width_tip, layer=layer_slab, name="slab"),)
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        sections=sections,
        **kwargs,
    )


@xsection
def strip_nitride_tip(
    width: float = 1.0,
    layer: typings.LayerSpec = "WGN",
    layer_silicon: typings.LayerSpec = "WG",
    width_tip_nitride: float = 0.2,
    width_tip_silicon: float = 0.1,
    radius: float = radius_nitride,
    radius_min: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return Strip cross_section."""
    sections = (
        Section(width=width_tip_nitride, layer=layer, name="tip_nitride"),
        Section(width=width_tip_silicon, layer=layer_silicon, name="tip_silicon"),
    )
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        sections=sections,
        **kwargs,
    )


@xsection
def slot(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    slot_width: float = 0.04,
    sections: Sections | None = None,
) -> CrossSection:
    """Return CrossSection Slot (with an etched region in the center).

    Args:
        width: main Section width (um) or function parameterized from 0 to 1. \
                the width at t==0 is the width at the beginning of the Path. \
                the width at t==1 is the width at the end.
        layer: main section layer.
        slot_width: in um.
        sections: list of Sections(width, offset, layer, ports).

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.slot(width=0.5, slot_width=0.05, layer='WG')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    rail_width = (width - slot_width) / 2
    rail_offset = (rail_width + slot_width) / 2

    section_list: list[Section] = list(sections or [])
    section_list.extend(
        [
            Section(
                width=rail_width, offset=rail_offset, layer=layer, name="left_rail"
            ),
            Section(
                width=rail_width, offset=-rail_offset, layer=layer, name="right rail"
            ),
        ]
    )

    return strip(
        width=width,
        layer="WG_ABSTRACT",
        sections=sections,
    )


@xsection
def rib_with_trenches(
    width: float = 0.5,
    width_trench: float = 2.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    simplify_slab: float | None = None,
    layer: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    wg_marking_layer: typings.LayerSpec = "WG_ABSTRACT",
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return CrossSection of rib waveguide defined by trenches.

    Args:
        width: main Section width (um) or function parameterized from 0 to 1. \
                the width at t==0 is the width at the beginning of the Path. \
                the width at t==1 is the width at the end.
        width_trench: in um.
        slab_offset: from the edge of the trench to the edge of the slab.
        width_slab: in um.
        simplify_slab: Optional Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting\
                polygon by more than the value listed here will be removed.
        layer: slab layer.
        layer_trench: layer to etch trenches.
        wg_marking_layer: layer to draw over the actual waveguide. \
                This can be useful for booleans, routing, placement ...
        sections: list of Sections(width, offset, layer, ports).
        kwargs: cross_section settings.

    .. code::

                        ┌─────────┐
                        │         │ wg_marking_layer
                        └─────────┘

               ┌────────┐         ┌────────┐
               │        │         │        │layer_trench
               └────────┘         └────────┘

         ┌─────────────────────────────────────────┐
         │                                  layer  │
         │                                         │
         └─────────────────────────────────────────┘
                        ◄─────────►
                           width
         ┌─────┐         ┌────────┐        ┌───────┐
         │     │         │        │        │       │
         │     └─────────┘        └────────┘       │
         │     ◄---------►         ◄-------►       │
         └─────────────────────────────────────────┘
                                            slab_offset
              width_trench                  ──────►
                                                   |
         ◄────────────────────────────────────────►
                      width_slab


    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.rib_with_trenches(width=0.5)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    if slab_offset is None and width_slab is None:
        raise ValueError("Must specify either slab_offset or width_slab")

    elif slab_offset is not None and width_slab is not None:
        raise ValueError("Cannot specify both slab_offset and width_slab")

    elif slab_offset is not None:
        width_slab = width + 2 * width_trench + 2 * slab_offset

    trench_offset = width / 2 + width_trench / 2
    section_list: list[Section] = list(sections or ())
    assert width_slab is not None
    section_list.append(
        Section(width=width_slab, layer=layer, name="slab", simplify=simplify_slab)
    )
    section_list += [
        Section(
            width=width_trench, offset=offset, layer=layer_trench, name=f"trench_{i}"
        )
        for i, offset in enumerate([+trench_offset, -trench_offset])
    ]

    return cross_section(
        layer=wg_marking_layer,
        width=width,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def l_with_trenches(
    width: float = 0.5,
    width_trench: float = 2.0,
    width_slab: float = 7.0,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    mirror: bool = False,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Return CrossSection of l waveguide defined by trenches.

    Args:
        width: main Section width (um) or function parameterized from 0 to 1. \
                the width at t==0 is the width at the beginning of the Path. \
                the width at t==1 is the width at the end.
        width_trench: in um.
        width_slab: in um.
        layer: ridge layer. None adds only ridge.
        layer_slab: slab layer.
        layer_trench: layer to etch trenches.
        mirror: this cross section is not symmetric and you can switch orientation.
        sections: list of Sections(width, offset, layer, ports).
        kwargs: cross_section settings.


    .. code::
                          x = 0
                           |
                           |
        _____         __________
             |        |         |
             |________|         |

       _________________________
             <------->          |
            width_trench
                       <-------->
                          width
                                |
       <------------------------>
            width_slab



    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.l_with_trenches(width=0.5)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    mult = 1 if mirror else -1
    trench_offset = mult * (width / 2 + width_trench / 2)
    section_list: list[Section] = list(sections or ())
    section_list += [
        Section(
            width=width_slab,
            layer=layer_slab,
            offset=mult * (width_slab / 2 - width / 2),
        )
    ]
    section_list += [
        Section(width=width_trench, offset=trench_offset, layer=layer_trench)
    ]

    return cross_section(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def metal1(
    width: float = 10,
    layer: typings.LayerSpec = "M1",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def metal2(
    width: float = 10,
    layer: typings.LayerSpec = "M2",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def metal3(
    width: float = 10,
    layer: typings.LayerSpec = "M3",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def metal_routing(
    width: float = 10,
    layer: typings.LayerSpec = "M3",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def heater_metal(
    width: float = 2.5,
    layer: typings.LayerSpec = "HEATER",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def npp(
    width: float = 0.5,
    layer: typings.LayerSpec = "NPP",
    radius: float | None = None,
    port_names: typings.IOPorts = port_names_electrical,
    port_types: typings.IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Doped NPP cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )


@xsection
def pin(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "SLAB90",
    layers_via_stack1: typings.LayerSpecs = ("PPP",),
    layers_via_stack2: typings.LayerSpecs = ("NPP",),
    bbox_offsets_via_stack1: tuple[float, ...] = (0, -0.2),
    bbox_offsets_via_stack2: tuple[float, ...] = (0, -0.2),
    via_stack_width: float = 9.0,
    via_stack_gap: float = 0.55,
    slab_gap: float = -0.2,
    layer_via: typings.LayerSpec | None = None,
    via_width: float = 1,
    via_offsets: tuple[float, ...] | None = None,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Rib PIN doped cross_section.

    Args:
        width: ridge width.
        layer: ridge layer.
        layer_slab: slab layer.
        layers_via_stack1: list of bot layer.
        layers_via_stack2: list of top layer.
        bbox_offsets_via_stack1: for bot.
        bbox_offsets_via_stack2: for top.
        via_stack_width: in um.
        via_stack_gap: offset from via_stack to ridge edge.
        slab_gap: extra slab gap (negative: via_stack goes beyond slab).
        layer_via: for via.
        via_width: in um.
        via_offsets: in um.
        sections: cross_section sections.
        kwargs: cross_section settings.


    https://doi.org/10.1364/OE.26.029983

    .. code::

                                      layer
                                |<----width--->|
                                 _______________ via_stack_gap           slab_gap
                                |              |<----------->|             <-->
        ___ ____________________|              |__________________________|___
       |   |         |                                       |            |   |
       |   |    P++  |         undoped silicon               |     N++    |   |
       |___|_________|_______________________________________|____________|___|
                                                              <----------->
                                                              via_stack_width
       <---------------------------------------------------------------------->
                                   slab_width

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.pin(width=0.5, via_stack_gap=1, via_stack_width=1)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    section_list: list[Section] = list(sections or [])
    slab_width = width + 2 * via_stack_gap + 2 * via_stack_width - 2 * slab_gap
    via_stack_offset = width / 2 + via_stack_gap + via_stack_width / 2

    section_list += [Section(width=slab_width, layer=layer_slab, name="slab")]
    section_list += [
        Section(
            layer=layer,
            width=via_stack_width + 2 * cladding_offset,
            offset=+via_stack_offset,
        )
        for layer, cladding_offset in zip(layers_via_stack1, bbox_offsets_via_stack1)
    ]
    section_list += [
        Section(
            layer=layer,
            width=via_stack_width + 2 * cladding_offset,
            offset=-via_stack_offset,
        )
        for layer, cladding_offset in zip(layers_via_stack2, bbox_offsets_via_stack2)
    ]
    if layer_via and via_width and via_offsets:
        section_list += [
            Section(
                layer=layer_via,
                width=via_width,
                offset=offset,
            )
            for offset in via_offsets
        ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def pn(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_slab: typings.LayerSpec = "SLAB90",
    gap_low_doping: float = 0.0,
    gap_medium_doping: float = 0.5,
    gap_high_doping: float = 1.0,
    offset_low_doping: float = 0.0,
    width_doping: float = 8.0,
    width_slab: float = 7.0,
    layer_p: typings.LayerSpec | None = "P",
    layer_pp: typings.LayerSpec | None = "PP",
    layer_ppp: typings.LayerSpec | None = "PPP",
    layer_n: typings.LayerSpec | None = "N",
    layer_np: typings.LayerSpec | None = "NP",
    layer_npp: typings.LayerSpec | None = "NPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    width_metal: float = 1.0,
    port_names: tuple[str, str] = ("o1", "o2"),
    sections: Sections | None = None,
    cladding_layers: typings.LayerSpecs | None = None,
    cladding_offsets: typings.Floats | None = None,
    cladding_simplify: typings.Floats | None = None,
    slab_inset: float | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Rib PN doped cross_section.

    Args:
        width: width of the ridge in um.
        layer: ridge layer.
        layer_slab: slab layer.
        gap_low_doping: from waveguide center to low doping. Only used for PIN.
        gap_medium_doping: from waveguide center to medium doping. None removes it.
        gap_high_doping: from center to high doping. None removes it.
        offset_low_doping: from center to junction center.
        width_doping: in um.
        width_slab: in um.
        layer_p: p doping layer.
        layer_pp: p+ doping layer.
        layer_ppp: p++ doping layer.
        layer_n: n doping layer.
        layer_np: n+ doping layer.
        layer_npp: n++ doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        width_metal: metal width in um.
        port_names: input and output port names.
        sections: optional list of sections.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        cladding_simplify: Optional Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting\
                polygon by more than the value listed here will be removed.
        slab_inset: slab inset in um.
        kwargs: cross_section settings.

    .. code::

                              offset_low_doping
                                <------>
                               |       |
                              wg     junction
                            center   center
                               |       |
                 ______________|_______|______
                 |             |       |     |
        _________|             |       |     |_________________|
              P                |       |               N       |
          width_p              |       |            width_n    |
        <----------------------------->|<--------------------->|
                               |               |       N+      |
                               |               |    width_n    |
                               |               |<------------->|
                               |<------------->|
                               gap_medium_doping

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.pn(width=0.5, gap_low_doping=0, width_doping=2.)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    slab_insets_valid = (slab_inset, slab_inset) if slab_inset else None

    slab = Section(
        width=width_slab, offset=0, layer=layer_slab, insets=slab_insets_valid
    )

    section_list: list[Section] = list(sections or [])
    section_list += [slab]
    base_offset_low_doping = width_doping / 2 + gap_low_doping / 4
    width_low_doping = width_doping - gap_low_doping / 2

    if layer_n:
        n = Section(
            width=width_low_doping + offset_low_doping,
            offset=+base_offset_low_doping - offset_low_doping / 2,
            layer=layer_n,
        )
        section_list.append(n)
    if layer_p:
        p = Section(
            width=width_low_doping - offset_low_doping,
            offset=-base_offset_low_doping - offset_low_doping / 2,
            layer=layer_p,
        )
        section_list.append(p)

    width_medium_doping = width_doping - gap_medium_doping
    offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

    if layer_np is not None:
        np = Section(
            width=width_medium_doping,
            offset=+offset_medium_doping,
            layer=layer_np,
        )
        section_list.append(np)
    if layer_pp is not None:
        pp = Section(
            width=width_medium_doping,
            offset=-offset_medium_doping,
            layer=layer_pp,
        )
        section_list.append(pp)

    width_high_doping = width_doping - gap_high_doping
    offset_high_doping = width_high_doping / 2 + gap_high_doping
    if layer_npp is not None:
        npp = Section(
            width=width_high_doping, offset=+offset_high_doping, layer=layer_npp
        )
        section_list.append(npp)
    if layer_ppp is not None:
        ppp = Section(
            width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp
        )
        section_list.append(ppp)

    if layer_via is not None:
        offset = width_high_doping + gap_high_doping - width_via / 2
        via_top = Section(width=width_via, offset=+offset, layer=layer_via)
        via_bot = Section(width=width_via, offset=-offset, layer=layer_via)
        section_list.append(via_top)
        section_list.append(via_bot)

    if layer_metal is not None:
        offset = width_high_doping + gap_high_doping - width_metal / 2
        port_types = ("electrical", "electrical")
        metal_top = Section(
            width=width_via,
            offset=+offset,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_top", "e2_top"),
        )
        metal_bot = Section(
            width=width_via,
            offset=-offset,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_bot", "e2_bot"),
        )
        section_list.append(metal_top)
        section_list.append(metal_bot)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=tuple(section_list),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        cladding_simplify=cladding_simplify,
        **kwargs,
    )


@xsection
def pn_with_trenches(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    gap_low_doping: float = 0.0,
    gap_medium_doping: float | None = 0.5,
    gap_high_doping: float | None = 1.0,
    offset_low_doping: float = 0.0,
    width_doping: float = 8.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    width_trench: float = 2.0,
    layer_p: typings.LayerSpec | None = "P",
    layer_pp: typings.LayerSpec | None = "PP",
    layer_ppp: typings.LayerSpec | None = "PPP",
    layer_n: typings.LayerSpec | None = "N",
    layer_np: typings.LayerSpec | None = "NP",
    layer_npp: typings.LayerSpec | None = "NPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    width_metal: float = 1.0,
    port_names: typings.IOPorts = ("o1", "o2"),
    cladding_layers: typings.Layers | None = cladding_layers_optical,
    cladding_offsets: typings.Floats | None = cladding_offsets_optical,
    cladding_simplify: typings.Floats | None = cladding_simplify_optical,
    wg_marking_layer: typings.LayerSpec | None = None,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Rib PN doped cross_section.

    Args:
        width: width of the ridge in um.
        layer: ridge layer. None adds only ridge.
        layer_trench: layer to etch trenches.
        gap_low_doping: from waveguide center to low doping. Only used for PIN.
        gap_medium_doping: from waveguide center to medium doping. None removes it.
        gap_high_doping: from center to high doping. None removes it.
        offset_low_doping: from center to junction center.
        width_doping: in um.
        slab_offset: from the edge of the trench to the edge of the slab.
        width_slab: in um.
        width_trench: in um.
        layer_p: p doping layer.
        layer_pp: p+ doping layer.
        layer_ppp: p++ doping layer.
        layer_n: n doping layer.
        layer_np: n+ doping layer.
        layer_npp: n++ doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        width_metal: metal width in um.
        port_names: input and output port names.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        cladding_simplify: Optional Tolerance value for the simplification algorithm.\
                All points that can be removed without changing the resulting. \
                polygon by more than the value listed here will be removed.
        wg_marking_layer: layer to draw over the actual waveguide.
        sections: optional list of sections.
        kwargs: cross_section settings.

    .. code::

                                   offset_low_doping
                                     <------>
                                    |       |
                                   wg     junction
                                 center   center             slab_offset
                                    |       |               <------>
        _____         ______________|_______ ______         ________
             |        |             |       |     |         |       |
             |________|             |             |_________|       |
                   P                |       |               N       |
               width_p              |                    width_n    |
          <-------------------------------->|<--------------------->|
             <------->              |               |       N+      |
            width_trench            |               |    width_n    |
                                    |               |<------------->|
                                    |<------------->|
                                    gap_medium_doping
       <------------------------------------------------------------>
                                width_slab

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.pn_with_trenches(width=0.5, gap_low_doping=0, width_doping=2.)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    if slab_offset is None and width_slab is None:
        raise ValueError("Must specify either slab_offset or width_slab")

    elif slab_offset is not None and width_slab is not None:
        raise ValueError("Cannot specify both slab_offset and width_slab")

    elif slab_offset is not None:
        width_slab = width + 2 * width_trench + 2 * slab_offset

    trench_offset = width / 2 + width_trench / 2
    section_list: list[Section] = list(sections or [])
    assert width_slab is not None
    section_list += [Section(width=width_slab, layer=layer)]
    section_list += [
        Section(width=width_trench, offset=offset, layer=layer_trench)
        for offset in [+trench_offset, -trench_offset]
    ]

    if wg_marking_layer is not None:
        section_list += [Section(width=width, offset=0, layer=wg_marking_layer)]

    base_offset_low_doping = width_doping / 2 + gap_low_doping / 4
    width_low_doping = width_doping - gap_low_doping / 2

    if layer_n:
        n = Section(
            width=width_low_doping + offset_low_doping,
            offset=+base_offset_low_doping - offset_low_doping / 2,
            layer=layer_n,
        )
        section_list.append(n)
    if layer_p:
        p = Section(
            width=width_low_doping - offset_low_doping,
            offset=-base_offset_low_doping - offset_low_doping / 2,
            layer=layer_p,
        )
        section_list.append(p)

    if gap_medium_doping is not None:
        width_medium_doping = width_doping - gap_medium_doping
        offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

        if layer_np:
            np = Section(
                width=width_medium_doping,
                offset=+offset_medium_doping,
                layer=layer_np,
            )
            section_list.append(np)
        if layer_pp:
            pp = Section(
                width=width_medium_doping,
                offset=-offset_medium_doping,
                layer=layer_pp,
            )
            section_list.append(pp)

    width_high_doping: float | None = None
    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width_high_doping / 2 + gap_high_doping
        if layer_npp:
            npp = Section(
                width=width_high_doping, offset=+offset_high_doping, layer=layer_npp
            )
            section_list.append(npp)
        if layer_ppp:
            ppp = Section(
                width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp
            )
            section_list.append(ppp)

    if (
        layer_via is not None
        and gap_high_doping is not None
        and width_high_doping is not None
    ):
        offset = width_high_doping + gap_high_doping - width_via / 2
        via_top = Section(width=width_via, offset=+offset, layer=layer_via)
        via_bot = Section(width=width_via, offset=-offset, layer=layer_via)
        section_list.append(via_top)
        section_list.append(via_bot)

    if (
        layer_metal is not None
        and width_high_doping is not None
        and gap_high_doping is not None
    ):
        offset = width_high_doping + gap_high_doping - width_metal / 2
        port_types = ("electrical", "electrical")
        metal_top = Section(
            width=width_via,
            offset=+offset,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_top", "e2_top"),
        )
        metal_bot = Section(
            width=width_via,
            offset=-offset,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_bot", "e2_bot"),
        )
        section_list.append(metal_top)
        section_list.append(metal_bot)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=tuple(section_list),
        cladding_offsets=cladding_offsets,
        cladding_simplify=cladding_simplify,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def pn_with_trenches_asymmetric(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    gap_low_doping: float | tuple[float, float] = (0.0, 0.0),
    gap_medium_doping: float | tuple[float, float] | None = (0.5, 0.2),
    gap_high_doping: float | tuple[float, float] | None = (1.0, 0.8),
    width_doping: float = 8.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    width_trench: float = 2.0,
    layer_p: typings.LayerSpec | None = "P",
    layer_pp: typings.LayerSpec | None = "PP",
    layer_ppp: typings.LayerSpec | None = "PPP",
    layer_n: typings.LayerSpec | None = "N",
    layer_np: typings.LayerSpec | None = "NP",
    layer_npp: typings.LayerSpec | None = "NPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    width_metal: float = 1.0,
    port_names: tuple[str, str] = ("o1", "o2"),
    cladding_layers: typings.Layers | None = cladding_layers_optical,
    cladding_offsets: typings.Floats | None = cladding_offsets_optical,
    wg_marking_layer: typings.LayerSpec | None = None,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Rib PN doped cross_section with asymmetric dimensions left and right.

    Args:
        width: width of the ridge in um.
        layer: ridge layer. None adds only ridge.
        layer_trench: layer to etch trenches.
        gap_low_doping: from waveguide center to low doping. Only used for PIN. \
                If a list, it considers the first element is [p_side, n_side]. If a number, \
                it assumes the same for both sides.
        gap_medium_doping: from waveguide center to medium doping. None removes it. \
                If a list, it considers the first element is [p_side, n_side]. \
                If a number, it assumes the same for both sides.
        gap_high_doping: from center to high doping. None removes it. \
                If a list, it considers the first element is [p_side, n_side].\
                If a number, it assumes the same for both sides.
        width_doping: in um.
        slab_offset: from the edge of the trench to the edge of the slab.
        width_slab: in um.
        width_trench: in um.
        layer_p: p doping layer.
        layer_pp: p+ doping layer.
        layer_ppp: p++ doping layer.
        layer_n: n doping layer.
        layer_np: n+ doping layer.
        layer_npp: n++ doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        width_metal: metal width in um.
        port_names: input and output port names.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        wg_marking_layer: layer to draw over the actual waveguide.
        sections: optional list of sections.
        kwargs: cross_section settings.

    .. code::

                                   gap_low_doping[1]
                                     <------>
                                    |       |
                                   wg     junction
                                 center   center           slab_offset
                                    |       |               <------>
        _____         ______________|_______ ______         ________
             |        |             |       |     |         |       |
             |________|             |             |_________|       |
                   P                |       |               N       |
               width_p              |                    width_n    |
          <-------------------------------->|<--------------------->|
             <------->              |               |       N+      |
            width_trench            |               |    width_n    |
                                    |               |<------------->|
                                    |<------------->|
                                    gap_medium_doping[1]
       <------------------------------------------------------------>
                                width_slab

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.pn_with_trenches_assymmetric(width=0.5, gap_low_doping=0, width_doping=2.)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    if slab_offset is None and width_slab is None:
        raise ValueError("Must specify either slab_offset or width_slab")

    elif slab_offset is not None and width_slab is not None:
        raise ValueError("Cannot specify both slab_offset and width_slab")

    elif slab_offset is not None:
        width_slab = width + 2 * width_trench + 2 * slab_offset

    # Trenches
    trench_offset = width / 2 + width_trench / 2
    section_list: list[Section] = list(sections or [])
    assert width_slab is not None
    section_list += [Section(width=width_slab, layer=layer)]
    section_list += [
        Section(width=width_trench, offset=offset, layer=layer_trench)
        for offset in [+trench_offset, -trench_offset]
    ]

    if wg_marking_layer is not None:
        section_list += [Section(width=width, offset=0, layer=wg_marking_layer)]

    # Low doping

    if not isinstance(gap_low_doping, list | tuple):
        gap_low_doping_list = [gap_low_doping] * 2
    else:
        gap_low_doping_list = list(gap_low_doping)

    if layer_n:
        width_low_doping_n = width_doping - gap_low_doping_list[1]
        n = Section(
            width=width_low_doping_n,
            offset=width_low_doping_n / 2 + gap_low_doping_list[1],
            layer=layer_n,
        )
        section_list.append(n)
    if layer_p:
        width_low_doping_p = width_doping - gap_low_doping_list[0]
        p = Section(
            width=width_low_doping_p,
            offset=-(width_low_doping_p / 2 + gap_low_doping_list[0]),
            layer=layer_p,
        )
        section_list.append(p)

    if gap_medium_doping is not None:
        if not isinstance(gap_medium_doping, list | tuple):
            gap_medium_doping_list = [gap_medium_doping] * 2
        else:
            gap_medium_doping_list = list(gap_medium_doping)

        if layer_np:
            width_np = width_doping - gap_medium_doping_list[1]
            np = Section(
                width=width_np,
                offset=width_np / 2 + gap_medium_doping_list[1],
                layer=layer_np,
            )
            section_list.append(np)
        if layer_pp:
            width_pp = width_doping - gap_medium_doping_list[0]
            pp = Section(
                width=width_pp,
                offset=-(width_pp / 2 + gap_medium_doping_list[0]),
                layer=layer_pp,
            )
            section_list.append(pp)
    gap_high_doping_list: list[float] | None = None
    width_npp: float | None = None
    width_ppp: float | None = None
    if gap_high_doping is not None:
        if not isinstance(gap_high_doping, list | tuple):
            gap_high_doping_list = [float(gap_high_doping)] * 2
        else:
            gap_high_doping_list = list(gap_high_doping)

        if layer_npp:
            width_npp = width_doping - gap_high_doping_list[1]
            npp = Section(
                width=width_npp,
                offset=width_npp / 2 + gap_high_doping_list[1],
                layer=layer_npp,
            )
            section_list.append(npp)
        if layer_ppp:
            width_ppp = width_doping - gap_high_doping_list[0]
            ppp = Section(
                width=width_ppp,
                offset=-(width_ppp / 2 + gap_high_doping_list[0]),
                layer=layer_ppp,
            )
            section_list.append(ppp)

    if (
        layer_via is not None
        and gap_high_doping_list is not None
        and width_npp is not None
        and width_ppp is not None
    ):
        offset_top = width_npp + gap_high_doping_list[1] - width_via / 2
        offset_bot = width_ppp + gap_high_doping_list[0] - width_via / 2
        via_top = Section(width=width_via, offset=+offset_top, layer=layer_via)
        via_bot = Section(width=width_via, offset=-offset_bot, layer=layer_via)
        section_list.append(via_top)
        section_list.append(via_bot)

    if (
        layer_metal is not None
        and gap_high_doping_list is not None
        and width_npp is not None
        and width_ppp is not None
    ):
        offset_top = width_npp + gap_high_doping_list[1] - width_metal / 2
        offset_bot = width_ppp + gap_high_doping_list[0] - width_metal / 2
        port_types = ("electrical", "electrical")
        metal_top = Section(
            width=width_via,
            offset=offset_top,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_top", "e2_top"),
        )
        metal_bot = Section(
            width=width_via,
            offset=-offset_bot,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_bot", "e2_bot"),
        )
        section_list.append(metal_top)
        section_list.append(metal_bot)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=tuple(section_list),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def l_wg_doped_with_trenches(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    layer_trench: typings.LayerSpec = "DEEP_ETCH",
    gap_low_doping: float = 0.0,
    gap_medium_doping: float | None = 0.5,
    gap_high_doping: float | None = 1.0,
    width_doping: float = 8.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    width_trench: float = 2.0,
    layer_low: typings.LayerSpec = "P",
    layer_mid: typings.LayerSpec = "PP",
    layer_high: typings.LayerSpec = "PPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    width_metal: float = 1.0,
    port_names: tuple[str, str] = ("o1", "o2"),
    cladding_layers: typings.Layers | None = cladding_layers_optical,
    cladding_offsets: typings.Floats | None = cladding_offsets_optical,
    wg_marking_layer: typings.LayerSpec | None = None,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """L waveguide PN doped cross_section.

    Args:
        width: width of the ridge in um.
        layer: ridge layer. None adds only ridge.
        layer_trench: layer to etch trenches.
        gap_low_doping: from waveguide outer edge to low doping. Only used for PIN.
        gap_medium_doping: from waveguide edge to medium doping. None removes it.
        gap_high_doping: from edge to high doping. None removes it.
        width_doping: in um.
        slab_offset: from the edge of the trench to the edge of the slab.
        width_slab: in um.
        width_trench: in um.
        layer_low: low doping layer.
        layer_mid: mid doping layer.
        layer_high: high doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        width_metal: metal width in um.
        port_names: input and output port names.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        wg_marking_layer: layer to mark where the actual guiding section is.
        sections: optional list of sections.
        kwargs: cross_section settings.

    .. code::

                                          gap_low_doping
                                           <------>
                                                  |
                                                  wg
                                                 edge
                                                  |
        _____                       _______ ______
             |                     |              |
             |_____________________|              |
                                                  |
                                                  |
                                    <------------>
                                           width
             <--------------------->               |
            width_trench       |                   |
                               |                   |
                               |<----------------->|
                                  gap_medium_doping
                     |<--------------------------->|
                             gap_high_doping
       <------------------------------------------->
                        width_slab

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.pn_with_trenches(width=0.5, gap_low_doping=0, width_doping=2.)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    if slab_offset is None and width_slab is None:
        raise ValueError("Must specify either slab_offset or width_slab")

    elif slab_offset is not None and width_slab is not None:
        raise ValueError("Cannot specify both slab_offset and width_slab")

    elif slab_offset is not None:
        width_slab = width + 2 * width_trench + 2 * slab_offset

    trench_offset = -1 * (width / 2 + width_trench / 2)
    section_list: list[Section] = list(sections or [])
    assert width_slab is not None
    section_list.append(
        Section(width=width_slab, layer=layer, offset=-1 * (width_slab / 2 - width / 2))
    )
    section_list += [
        Section(width=width_trench, offset=trench_offset, layer=layer_trench)
    ]

    if wg_marking_layer is not None:
        section_list += [Section(width=width, offset=0, layer=wg_marking_layer)]

    offset_low_doping = width / 2 - gap_low_doping - width_doping / 2

    low_doping = Section(
        width=width_doping,
        offset=offset_low_doping,
        layer=layer_low,
    )

    section_list.append(low_doping)

    if gap_medium_doping is not None:
        width_medium_doping = width_doping - gap_medium_doping
        offset_medium_doping = width / 2 - gap_medium_doping - width_medium_doping / 2

        mid_doping = Section(
            width=width_medium_doping,
            offset=offset_medium_doping,
            layer=layer_mid,
        )
        section_list.append(mid_doping)

    offset_high_doping: float | None = None
    width_high_doping: float | None = None

    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width / 2 - gap_high_doping - width_high_doping / 2

        high_doping = Section(
            width=width_high_doping, offset=+offset_high_doping, layer=layer_high
        )

        section_list.append(high_doping)

    if (
        layer_via is not None
        and offset_high_doping is not None
        and width_high_doping is not None
    ):
        offset = offset_high_doping - width_high_doping / 2 + width_via / 2
        via = Section(width=width_via, offset=+offset, layer=layer_via)
        section_list.append(via)

    if (
        layer_metal is not None
        and offset_high_doping is not None
        and width_high_doping is not None
    ):
        offset = offset_high_doping - width_high_doping / 2 + width_metal / 2
        port_types = ("electrical", "electrical")
        metal = Section(
            width=width_via,
            offset=+offset,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_top", "e2_top"),
        )
        section_list.append(metal)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=tuple(section_list),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def strip_heater_metal_undercut(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 2.5,
    trench_width: float = 6.5,
    trench_gap: float = 2.0,
    layer_heater: typings.LayerSpec = "HEATER",
    layer_trench: typings.LayerSpec = "DEEPTRENCH",
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns strip cross_section with top metal and undercut trenches on both.

    sides.

    dimensions from https://doi.org/10.1364/OE.18.020298

    Args:
        width: waveguide width.
        layer: waveguide layer.
        heater_width: of metal heater.
        trench_width: in um.
        trench_gap: from waveguide edge to trench edge.
        layer_heater: heater layer.
        layer_trench: tench layer.
        sections: cross_section sections.
        kwargs: cross_section settings.

    .. code::

              |<-------heater_width--------->|
               ______________________________
              |                              |
              |         layer_heater         |
              |______________________________|

                   |<------width------>|
                    ____________________ trench_gap
                   |                   |<----------->|              |
                   |                   |             |   undercut   |
                   |       width       |             |              |
                   |                   |             |<------------>|
                   |___________________|             | trench_width |
                                                     |              |
                                                     |              |

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.strip_heater_metal_undercut(width=0.5, heater_width=2, trench_width=4, trench_gap=4)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    trench_offset = trench_gap + trench_width / 2 + width / 2
    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(
            layer=layer_heater,
            width=heater_width,
            port_names=port_names_electrical,
            port_types=port_types_electrical,
        ),
        Section(layer=layer_trench, width=trench_width, offset=+trench_offset),
        Section(layer=layer_trench, width=trench_width, offset=-trench_offset),
    ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def strip_heater_metal(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 2.5,
    layer_heater: typings.LayerSpec = "HEATER",
    sections: Sections | None = None,
    insets: tuple[float, float] | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns strip cross_section with top heater metal.

    dimensions from https://doi.org/10.1364/OE.18.020298

    Args:
        width: waveguide width (um).
        layer: waveguide layer.
        heater_width: of metal heater.
        layer_heater: for the metal.
        sections: cross_section sections.
        insets: for the heater.
        kwargs: cross_section settings.

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.strip_heater_metal(width=0.5, heater_width=2)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(
            layer=layer_heater,
            width=heater_width,
            port_names=port_names_electrical,
            port_types=port_types_electrical,
            insets=insets,
        )
    ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def strip_heater_doped(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    layers_heater: typings.LayerSpecs = ("WG", "NPP"),
    bbox_offsets_heater: tuple[float, ...] = (0, 0.1),
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns strip cross_section with N++ doped heaters on both sides.

    Args:
        width: in um.
        layer: waveguide spec.
        heater_width: in um.
        heater_gap: in um.
        layers_heater: for doped heater.
        bbox_offsets_heater: for each layers_heater.
        sections: cross_section sections.
        kwargs: cross_section settings.

    .. code::

                                  |<------width------>|
          ____________             ___________________               ______________
         |            |           |     undoped Si    |             |              |
         |layer_heater|           |  intrinsic region |<----------->| layer_heater |
         |____________|           |___________________|             |______________|
                                                                     <------------>
                                                        heater_gap     heater_width

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.strip_heater_doped(width=0.5, heater_width=2, heater_gap=0.5)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    heater_offset = width / 2 + heater_gap + heater_width / 2

    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=+heater_offset,
            name=f"heater_upper_{layer}",
        )
        for layer, cladding_offset in zip(layers_heater, bbox_offsets_heater)
    ]

    section_list += [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=-heater_offset,
            name=f"heater_lower_{layer}",
        )
        for layer, cladding_offset in zip(layers_heater, bbox_offsets_heater)
    ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def rib_heater_doped(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    layer_heater: typings.LayerSpec = "NPP",
    layer_slab: typings.LayerSpec = "SLAB90",
    slab_gap: float = 0.2,
    with_top_heater: bool = True,
    with_bot_heater: bool = True,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns rib cross_section with N++ doped heaters on both sides.

    dimensions from https://doi.org/10.1364/OE.27.010456

    .. code::

                                    |<------width------>|
                                     ____________________  heater_gap           slab_gap
                                    |                   |<----------->|             <-->
         ___ _______________________|                   |__________________________|___
        |   |            |                undoped Si                  |            |   |
        |   |layer_heater|                intrinsic region            |layer_heater|   |
        |___|____________|____________________________________________|____________|___|
                                                                       <---------->
                                                                        heater_width
        <------------------------------------------------------------------------------>
                                        slab_width

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.rib_heater_doped(width=0.5, heater_width=2, heater_gap=0.5, layer_heater='NPP')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    heater_offset = width / 2 + heater_gap + heater_width / 2

    if with_bot_heater and with_top_heater:
        slab_width = width + 2 * heater_gap + 2 * heater_width + 2 * slab_gap
        slab_offset = 0.0
    elif with_top_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = -slab_width / 2
    elif with_bot_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = +slab_width / 2
    else:
        raise ValueError("At least one heater must be True")

    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(width=slab_width, layer=layer_slab, offset=slab_offset, name="slab")
    ]

    if with_bot_heater:
        section_list += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=+heater_offset,
                name="heater_upper",
            )
        ]
    if with_top_heater:
        section_list += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=-heater_offset,
                name="heater_lower",
            )
        ]
    return strip(
        width=width,
        layer=layer,
        sections=tuple(section_list),
        **kwargs,
    )


@xsection
def rib_heater_doped_via_stack(
    width: float = 0.5,
    layer: typings.LayerSpec = "WG",
    heater_width: float = 1.0,
    heater_gap: float = 0.8,
    layer_slab: typings.LayerSpec = "SLAB90",
    layer_heater: typings.LayerSpec = "NPP",
    via_stack_width: float = 2.0,
    via_stack_gap: float = 0.8,
    layers_via_stack: typings.LayerSpecs = ("NPP", "VIAC"),
    bbox_offsets_via_stack: tuple[float, ...] = (0, -0.2),
    slab_gap: float = 0.2,
    slab_offset: float = 0,
    with_top_heater: bool = True,
    with_bot_heater: bool = True,
    sections: Sections | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Returns rib cross_section with N++ doped heaters on both sides.

    dimensions from https://doi.org/10.1364/OE.27.010456

    Args:
        width: in um.
        layer: for main waveguide section.
        heater_width: in um.
        heater_gap: in um.
        layer_slab: for pedestal.
        layer_heater: for doped heater.
        via_stack_width: for the contact.
        via_stack_gap: in um.
        layers_via_stack: for the contact.
        bbox_offsets_via_stack: for the contact.
        slab_gap: from heater edge.
        slab_offset: over the center of the slab.
        with_top_heater: adds top/left heater.
        with_bot_heater: adds bottom/right heater.
        sections: list of sections to add to the cross_section.
        kwargs: cross_section settings.

    .. code::

                                   |<----width------>|
       slab_gap                     __________________ via_stack_gap     via_stack width
       <-->                        |                 |<------------>|<--------------->
                                   |                 | heater_gap |
                                   |                 |<---------->|
        ___ _______________________|                 |___________________________ ____
       |   |            |              undoped Si                 |              |    |
       |   |layer_heater|              intrinsic region           |layer_heater  |    |
       |___|____________|_________________________________________|______________|____|
                                                                   <------------>
                                                                    heater_width
       <------------------------------------------------------------------------------>
                                       slab_width

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.rib_heater_doped_via_stack(width=0.5, heater_width=2, heater_gap=0.5, layer_heater='NPP')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    if with_bot_heater and with_top_heater:
        slab_width = width + 2 * heater_gap + 2 * heater_width + 2 * slab_gap
    elif with_top_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset -= slab_width / 2
    elif with_bot_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset += slab_width / 2
    else:
        raise ValueError("At least one heater must be True")

    heater_offset = width / 2 + heater_gap + heater_width / 2
    via_stack_offset = width / 2 + via_stack_gap + via_stack_width / 2
    section_list: list[Section] = list(sections or [])
    section_list += [
        Section(width=slab_width, layer=layer_slab, offset=slab_offset, name="slab"),
    ]
    if with_bot_heater:
        section_list += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=+heater_offset,
            )
        ]

    if with_top_heater:
        section_list += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=-heater_offset,
            )
        ]

    if with_bot_heater:
        section_list += [
            Section(
                layer=layer,
                width=heater_width + 2 * cladding_offset,
                offset=+via_stack_offset,
            )
            for layer, cladding_offset in zip(layers_via_stack, bbox_offsets_via_stack)
        ]

    if with_top_heater:
        section_list += [
            Section(
                layer=layer,
                width=heater_width + 2 * cladding_offset,
                offset=-via_stack_offset,
            )
            for layer, cladding_offset in zip(layers_via_stack, bbox_offsets_via_stack)
        ]

    return strip(
        sections=tuple(section_list),
        width=width,
        layer=layer,
        **kwargs,
    )


@xsection
def pn_ge_detector_si_contacts(
    width_si: float = 6.0,
    layer_si: typings.LayerSpec = "WG",
    width_ge: float = 3.0,
    layer_ge: typings.LayerSpec = "GE",
    gap_low_doping: float = 0.6,
    gap_medium_doping: float = 0.9,
    gap_high_doping: float = 1.1,
    width_doping: float = 8.0,
    layer_p: typings.LayerSpec = "P",
    layer_pp: typings.LayerSpec = "PP",
    layer_ppp: typings.LayerSpec = "PPP",
    layer_n: typings.LayerSpec = "N",
    layer_np: typings.LayerSpec = "NP",
    layer_npp: typings.LayerSpec = "NPP",
    layer_via: typings.LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: typings.LayerSpec | None = None,
    port_names: tuple[str, str] = ("o1", "o2"),
    cladding_layers: typings.Layers | None = cladding_layers_optical,
    cladding_offsets: typings.Floats | None = cladding_offsets_optical,
    cladding_simplify: typings.Floats | None = None,
    **kwargs: Any,
) -> CrossSection:
    """Linear Ge detector cross section based on a lateral p(i)n junction.

    It has silicon contacts (no contact on the Ge). The contacts need to be
    created in the component generating function (they can't be created here).

    See Chen et al., "High-Responsivity Low-Voltage 28-Gb/s Ge p-i-n Photodetector
    With Silicon Contacts", Journal of Lightwave Technology 33(4), 2015.

    Notice it is possible to have dopings going beyond the ridge waveguide. This
    is fine, and it is to account for the
    presence of the contacts. Such contacts can be subwavelength or not.

    Args:
        width_si: width of the full etch si in um.
        layer_si: si ridge layer.
        width_ge: width of the ge in um.
        layer_ge: ge layer.
        gap_low_doping: from waveguide center to low doping.
        gap_medium_doping: from waveguide center to medium doping. None removes it.
        gap_high_doping: from center to high doping. None removes it.
        width_doping: distance from waveguide center to doping edge in um.
        layer_p: p doping layer.
        layer_pp: p+ doping layer.
        layer_ppp: p++ doping layer.
        layer_n: n doping layer.
        layer_np: n+ doping layer.
        layer_npp: n++ doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        port_names: for input and output ('o1', 'o2').
        cladding_layers: list of layers to extrude.
        cladding_offsets: list of offset from main Section edge.
        cladding_simplify: Optional Tolerance value for the simplification algorithm. \
                All points that can be removed without changing the resulting. \
                polygon by more than the value listed here will be removed.
        kwargs: cross_section settings.

    .. code::

                                   layer_si
                           |<------width_si---->|

                                  layer_ge
                              |<--width_ge->|
                               ______________
                              |             |
                            __|_____________|___
                           |     |       |     |
                           |     |       |     |
                    P      |     |       |     |         N                |
                 width_p   |_____|_______|_____|           width_n        |
        <----------------------->|       |<------------------------------>|
                                     |<->|
                                     gap_low_doping
                                     |         |        N+                |
                                     |         |     width_np             |
                                     |         |<------------------------>|
                                     |<------->|
                                     |     gap_medium_doping
                                     |
                                     |<---------------------------------->|
                                                width_doping

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.pn()
        p = gf.path.straight()
        c = p.extrude(xs)
        c.plot()
    """
    width_low_doping = width_doping - gap_low_doping
    offset_low_doping = width_low_doping / 2 + gap_low_doping

    s = Section(width=width_si, offset=0, layer=layer_si, port_names=port_names)
    n = Section(width=width_low_doping, offset=+offset_low_doping, layer=layer_n)
    p = Section(width=width_low_doping, offset=-offset_low_doping, layer=layer_p)

    section_list = [s, n, p]

    cladding_layers = cladding_layers or ()
    cladding_offsets = cladding_offsets or ()
    cladding_simplify_not_none = cladding_simplify or (None,) * len(cladding_layers)
    section_list += [
        Section(width=width_si + 2 * offset, layer=layer, simplify=simplify)
        for layer, offset, simplify in zip(
            cladding_layers, cladding_offsets, cladding_simplify_not_none
        )
    ]

    width_medium_doping = width_doping - gap_medium_doping
    offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

    np = Section(
        width=width_medium_doping,
        offset=+offset_medium_doping,
        layer=layer_np,
    )
    pp = Section(
        width=width_medium_doping,
        offset=-offset_medium_doping,
        layer=layer_pp,
    )
    section_list.extend((np, pp))
    width_high_doping = width_doping - gap_high_doping
    offset_high_doping = width_high_doping / 2 + gap_high_doping
    npp = Section(width=width_high_doping, offset=+offset_high_doping, layer=layer_npp)
    ppp = Section(width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp)
    section_list.extend((npp, ppp))
    if layer_via is not None:
        offset = width_high_doping / 2 + gap_high_doping
        via_top = Section(width=width_via, offset=+offset, layer=layer_via)
        via_bot = Section(width=width_via, offset=-offset, layer=layer_via)
        section_list.extend((via_top, via_bot))
    if layer_metal is not None:
        offset = width_high_doping / 2 + gap_high_doping
        port_types = ("electrical", "electrical")
        metal_top = Section(
            width=width_via,
            offset=+offset,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_top", "e2_top"),
        )
        metal_bot = Section(
            width=width_via,
            offset=-offset,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_bot", "e2_bot"),
        )
        section_list.extend((metal_top, metal_bot))

    # Add the Ge
    s = Section(width=width_ge, offset=0, layer=layer_ge)
    section_list.append(s)

    return CrossSection(
        sections=tuple(section_list),
        **kwargs,
    )


def is_cross_section(name: str, obj: Any, verbose: bool = False) -> bool:
    """Check if an object is a cross-section factory function.

    Args:
        name: Name of the object.
        obj: Object to check.
        verbose: Whether to print warnings for errors.

    Returns:
        True if the object is a cross-section factory function.
    """
    if name.startswith("_"):
        return False

    # Early prune: only consider functions, builtins or partials
    func: FunctionType | BuiltinFunctionType | None = None
    if isfunction(obj) or isbuiltin(obj):
        func = obj
    elif isinstance(obj, partial):
        # Check if the underlying function is a function or builtin
        if isfunction(obj.func) or isbuiltin(obj.func):
            func = obj.func
        else:
            return False
    else:
        return False

    # Ensure func is not None for type checker
    if func is None:
        return False

    # Check if function is registered in the cross_sections dictionary
    # This happens when decorated with @xsection
    if name in cross_sections and cross_sections[name] is obj:
        return True

    # Fallback: check return type annotation
    try:
        ann = getattr(func, "__annotations__", {})
        return_type = ann.get("return")

        if return_type is None:
            return False

        # Handle string annotations and forward references
        if isinstance(return_type, str):
            # Handle simple string matches
            if return_type in (
                "CrossSection",
                "gf.CrossSection",
                "gdsfactory.CrossSection",
            ):
                return True

            # For other string annotations, try to resolve them in the function's context
            try:
                # Try globals first
                func_globals = getattr(func, "__globals__", {})
                resolved_type = func_globals.get(return_type)

                # If not in globals, try closure variables
                if (
                    resolved_type is None
                    and hasattr(func, "__closure__")
                    and func.__closure__
                ):
                    # Get the names of closure variables
                    if hasattr(func, "__code__") and hasattr(
                        func.__code__, "co_freevars"
                    ):
                        freevars = func.__code__.co_freevars
                        closure_values = func.__closure__
                        if len(freevars) == len(closure_values):
                            closure_dict = dict(
                                zip(
                                    freevars,
                                    [cell.cell_contents for cell in closure_values],
                                )
                            )
                            resolved_type = closure_dict.get(return_type)

                if resolved_type and isinstance(resolved_type, type):
                    return issubclass(resolved_type, CrossSection)

            except (TypeError, AttributeError, ValueError):
                pass

            return False

        # Direct type comparison
        if return_type is CrossSection:
            return True

        # Check if it's a subclass of CrossSection
        if isinstance(return_type, type):
            try:
                return issubclass(return_type, CrossSection)
            except TypeError:
                # Handle cases where return_type is not a class
                return False

    except Exception as e:
        if verbose:
            logger.warning(f"Error checking cross-section for {name}: {e}")

    return False


def get_cross_sections(
    modules: Sequence[ModuleType] | ModuleType, verbose: bool = False
) -> dict[str, CrossSectionFactory]:
    """Returns cross_sections from a module or list of modules.

    Args:
        modules: module or iterable of modules.
        verbose: prints in case any errors occur.
    """
    # Optimize module input normalization and preallocate xs
    if isinstance(modules, Sequence) and not isinstance(modules, str):
        modules_ = modules
    else:
        modules_ = [modules]

    xs: dict[str, CrossSectionFactory] = {
        name: obj
        for module in modules_
        for name, obj in getmembers(module)
        if is_cross_section(name, obj, verbose)
    }

    return xs


# cross_sections = get_cross_sections(sys.modules[__name__])


if __name__ == "__main__":
    # xs = gf.cross_section.pn(
    #     # slab_offset=0
    #     # offset=1,
    #     # cladding_layers=[(2, 0)],
    #     # cladding_offsets=[3],
    #     # bbox_layers=[(3, 0)],
    #     # bbox_offsets=[2],
    #     # slab_inset=0.2,
    # )
    # xs = xs.append_sections(sections=[gf.Section(width=1.0, layer=(2, 0), name="slab")])
    # p = gf.path.straight()
    # c = p.extrude(xs)
    # c = gf.c.straight(cross_section=xs)
    # xs = pn(slab_inset=0.2)
    # xs = metal1()
    # s0 = Section(width=2, layer=(1, 0))
    # xs = strip()
    # print(xs.name)
    import gdsfactory as gf

    xs1 = gf.get_cross_section("metal_routing")

    xs2 = xs1.copy(width=10)
    assert xs2.name == xs1.name, f"{xs2.name} != {xs1.name}"
    print(xs2.name)
