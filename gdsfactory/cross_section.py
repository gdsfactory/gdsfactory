"""You can define a path as list of points.

To create a component you need to extrude the path with a cross-section.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable, Iterable
from functools import partial, wraps
from inspect import getmembers, signature
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from kfactory import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    model_validator,
)

from gdsfactory.config import CONF, ErrorType

if TYPE_CHECKING:
    from gdsfactory.typings import AnyComponentT, PortNames, PortTypes

nm = 1e-3

Layer: TypeAlias = tuple[int, int]
Layers: TypeAlias = tuple[Layer, ...]
WidthTypes: TypeAlias = Literal["sine", "linear", "parabolic"]

LayerSpec: TypeAlias = Layer | str
LayerSpecs: TypeAlias = Iterable[LayerSpec]

Floats: TypeAlias = tuple[float, ...]

WidthFunction: TypeAlias = Callable[..., npt.NDArray[np.float64]]
OffsetFunction: TypeAlias = Callable[..., npt.NDArray[np.float64]]

port_names_electrical: "PortNames" = ("e1", "e2")
port_types_electrical: "PortTypes" = ("electrical", "electrical")
cladding_layers_optical = None
cladding_offsets_optical = None
cladding_simplify_optical = None

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
    layer: LayerSpec | None = None
    port_names: tuple[str | None, str | None] = (None, None)
    port_types: tuple[str, str] = ("optical", "optical")
    name: str | None = None
    hidden: bool = False
    simplify: float | None = None

    width_function: WidthFunction | None = Field(default=None)
    offset_function: OffsetFunction | None = Field(default=None)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def generate_default_name(cls, data: Any) -> Any:
        if not data.get("name"):
            h = hashlib.md5(str(data).encode()).hexdigest()[:8]
            data["name"] = f"s_{h}"
        return data

    @field_serializer("width_function", "offset_function")
    def serialize_functions(self, func: Callable | None) -> str | None:
        if func is None:
            return None
        t_values = np.linspace(0, 1, 11)
        return ",".join([str(round(width, 3)) for width in func(t_values)])


class ComponentAlongPath(BaseModel):
    """A ComponentAlongPath object to place along an extruded path.

    Parameters:
        component: to repeat along the path. The unrotated version should be oriented \
                for placement on a horizontal line.
        spacing: distance between component placements
        padding: minimum distance from the path start to the first component.
        y_offset: offset in y direction (um).
    """

    component: object
    spacing: float
    padding: float = 0.0
    offset: float = 0.0

    model_config = ConfigDict(extra="forbid", frozen=True)


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

    sections: tuple[Section, ...] = Field(default_factory=tuple)
    components_along_path: tuple[ComponentAlongPath, ...] = Field(default_factory=tuple)
    radius: float | None = None
    radius_min: float | None = None
    bbox_layers: LayerSpecs | None = None
    bbox_offsets: Floats | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)
    _name = PrivateAttr("")

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
                warnings.warn(message)

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
    def layer(self) -> LayerSpec:
        return self.sections[0].layer

    def append_sections(self, sections: Sections) -> CrossSection:
        """Append sections to the cross_section."""
        sections = self.sections + tuple(sections)
        return self.model_copy(update={"sections": sections})

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
        layer: LayerSpec | None = None,
        width_function: WidthFunction | None = None,
        offset_function: OffsetFunction | None = None,
        sections: tuple[Section, ...] | None = None,
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
                sections = self.sections
            sections = [s.model_copy() for s in sections]
            sections[0] = sections[0].model_copy(
                update={
                    "width_function": width_function,
                    "offset_function": offset_function,
                    "width": width or self.width,
                    "layer": layer or self.layer,
                }
            )
            xs = self.model_copy(update={"sections": tuple(sections), **kwargs})
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

    #     kf.kcl.enclosure = kf.KCellEnclosure(
    #         enclosures=[enclosure_rc],
    #     )
    #     component.kcl.enclosure.apply_minkowski_y(component)

    def add_bbox(
        self,
        component: AnyComponentT,
        top: float | None = None,
        bottom: float | None = None,
        right: float | None = None,
        left: float | None = None,
    ) -> AnyComponentT:
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
            padding = []
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


CrossSectionSpec = CrossSection | str | dict[str, Any] | Callable[..., CrossSection]
CrossSectionFactory = Callable[..., CrossSection]
ConductorConductorName = tuple[str, str]
ConductorViaConductorName = tuple[str, str, str] | tuple[str, str]
ConnectivitySpec = ConductorConductorName | ConductorViaConductorName


class Transition(BaseModel):
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
    width_type: WidthTypes | Callable[[float, float, float], float] = "sine"
    offset_type: WidthTypes | Callable[[float, float, float], float] = "sine"

    @field_serializer("width_type")
    def serialize_width(
        self,
        width_type: WidthTypes
        | Callable[[npt.NDArray[np.float_], float, float], npt.NDArray[np.float_]],
    ) -> str:
        if isinstance(width_type, str):
            return width_type
        t_values = np.linspace(0, 1, 10)
        return ",".join(
            [str(round(width, 3)) for width in width_type(t_values, *self.width)]
        )


cross_sections = {}
_cross_section_default_names = {}


def xsection(func: Callable[..., CrossSection]) -> Callable[..., CrossSection]:
    """Decorator to register a cross section function.

    Ensures that the cross-section name matches the name of the function that generated it when created using default parameters

    .. code-block:: python

        @xsection
        def xs_sc(width=TECH.width_sc, radius=TECH.radius_sc):
            return gf.cross_section.cross_section(width=width, radius=radius)
    """
    default_xs = func()
    _cross_section_default_names[default_xs.name] = func.__name__

    @wraps(func)
    def newfunc(**kwargs: Any) -> CrossSection:
        xs = func(**kwargs)
        if xs.name in _cross_section_default_names:
            xs._name = _cross_section_default_names[xs.name]
        return xs

    cross_sections[func.__name__] = newfunc
    return newfunc


def cross_section(
    width: float = 0.5,
    offset: float = 0,
    layer: LayerSpec | None = "WG",
    sections: tuple[Section, ...] | None = None,
    port_names: tuple[str, str] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    bbox_layers: "LayerSpecs | None" = None,
    bbox_offsets: "Floats | None" = None,
    cladding_layers: "LayerSpecs | None" = None,
    cladding_offsets: "Floats | None" = None,
    cladding_simplify: "Floats | None" = None,
    cladding_centers: "Floats | None" = None,
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
    sections = list(sections or [])

    if cladding_layers:
        cladding_simplify = cladding_simplify or (None,) * len(cladding_layers)
        cladding_offsets = cladding_offsets or (0,) * len(cladding_layers)
        cladding_centers = cladding_centers or [0] * len(cladding_layers)

        if (
            len(
                {
                    len(x)
                    for x in (
                        cladding_layers,
                        cladding_offsets,
                        cladding_simplify,
                        cladding_centers,
                    )
                }
            )
            > 1
        ):
            raise ValueError(
                f"{len(cladding_layers)=}, "
                f"{len(cladding_offsets)=}, "
                f"{len(cladding_simplify)=}, "
                f"{len(cladding_centers)=} must have same length"
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
    ] + sections

    if cladding_layers:
        s += [
            Section(
                width=width + 2 * offset, layer=layer, simplify=simplify, offset=center
            )
            for layer, offset, simplify, center in zip(
                cladding_layers, cladding_offsets, cladding_simplify, cladding_centers
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
    layer: LayerSpec = "WG",
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
def rib(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    radius: float = radius_rib,
    radius_min: float | None = None,
    cladding_layers: LayerSpecs = ("SLAB90",),
    cladding_offsets: Floats = (3,),
    cladding_simplify: Floats = (50 * nm,),
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
    layer: LayerSpec = "WG",
    radius: float = radius_rib,
    radius_min: float | None = None,
    bbox_layers: LayerSpecs = ("SLAB90",),
    bbox_offsets: Floats = (3,),
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
    layer: LayerSpec = "WG",
    layer_slab: LayerSpec = "SLAB90",
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
    layer: LayerSpec = "WGN",
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
    layer: LayerSpec = "WG",
    layer_slab: LayerSpec = "SLAB90",
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
    layer: LayerSpec = "WGN",
    layer_silicon: LayerSpec = "WG",
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
    layer: LayerSpec = "WG",
    slot_width: float = 0.04,
    sections: tuple[Section, ...] | None = None,
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

    sections = sections or ()
    sections += (
        Section(width=rail_width, offset=rail_offset, layer=layer, name="left_rail"),
        Section(width=rail_width, offset=-rail_offset, layer=layer, name="right rail"),
    )

    return strip(
        width=width,
        layer=None,
        sections=sections,
    )


@xsection
def rib_with_trenches(
    width: float = 0.5,
    width_trench: float = 2.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    simplify_slab: float | None = None,
    layer: LayerSpec | None = "WG",
    layer_trench: LayerSpec = "DEEP_ETCH",
    wg_marking_layer: LayerSpec | None = None,
    sections: tuple[Section, ...] | None = None,
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
    sections = list(sections or ())
    sections += [
        Section(width=width_slab, layer=layer, name="slab", simplify=simplify_slab)
    ]
    sections += [
        Section(
            width=width_trench, offset=offset, layer=layer_trench, name=f"trench_{i}"
        )
        for i, offset in enumerate([+trench_offset, -trench_offset])
    ]

    return cross_section(
        layer=wg_marking_layer,
        width=width,
        sections=tuple(sections),
        **kwargs,
    )


@xsection
def l_with_trenches(
    width: float = 0.5,
    width_trench: float = 2.0,
    width_slab: float = 7.0,
    layer: LayerSpec | None = "WG",
    layer_slab: LayerSpec | None = "WG",
    layer_trench: LayerSpec = "DEEP_ETCH",
    mirror: bool = False,
    sections: tuple[Section, ...] | None = None,
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
    sections = list(sections or ())
    sections += [
        Section(
            width=width_slab,
            layer=layer_slab,
            offset=mult * (width_slab / 2 - width / 2),
        )
    ]
    sections += [Section(width=width_trench, offset=trench_offset, layer=layer_trench)]

    return cross_section(
        width=width,
        layer=layer,
        sections=tuple(sections),
        **kwargs,
    )


@xsection
def metal1(
    width: float = 10,
    layer: LayerSpec = "M1",
    radius: float | None = None,
    port_names: "PortNames" = port_names_electrical,
    port_types: "PortTypes" = port_types_electrical,
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
    layer: LayerSpec = "M2",
    radius: float | None = None,
    port_names: "PortNames" = port_names_electrical,
    port_types: "PortTypes" = port_types_electrical,
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
    layer: LayerSpec = "M3",
    radius: float | None = None,
    port_names: "PortNames" = port_names_electrical,
    port_types: "PortTypes" = port_types_electrical,
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
    layer: LayerSpec = "M3",
    radius: float | None = None,
    port_names: "PortNames" = port_names_electrical,
    port_types: "PortTypes" = port_types_electrical,
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
    layer: LayerSpec = "HEATER",
    radius: float | None = None,
    port_names: "PortNames" = port_names_electrical,
    port_types: "PortTypes" = port_types_electrical,
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
    layer: LayerSpec = "NPP",
    radius: float | None = None,
    port_names: "PortNames" = port_names_electrical,
    port_types: "PortTypes" = port_types_electrical,
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
    layer: LayerSpec = "WG",
    layer_slab: LayerSpec = "SLAB90",
    layers_via_stack1: LayerSpecs = ("PPP",),
    layers_via_stack2: LayerSpecs = ("NPP",),
    bbox_offsets_via_stack1: tuple[float, ...] = (0, -0.2),
    bbox_offsets_via_stack2: tuple[float, ...] = (0, -0.2),
    via_stack_width: float = 9.0,
    via_stack_gap: float = 0.55,
    slab_gap: float = -0.2,
    layer_via: LayerSpec | None = None,
    via_width: float = 1,
    via_offsets: tuple[float, ...] | None = None,
    sections: tuple[Section, ...] | None = None,
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
    sections = list(sections or [])
    slab_width = width + 2 * via_stack_gap + 2 * via_stack_width - 2 * slab_gap
    via_stack_offset = width / 2 + via_stack_gap + via_stack_width / 2

    sections += [Section(width=slab_width, layer=layer_slab, name="slab")]
    sections += [
        Section(
            layer=layer,
            width=via_stack_width + 2 * cladding_offset,
            offset=+via_stack_offset,
        )
        for layer, cladding_offset in zip(layers_via_stack1, bbox_offsets_via_stack1)
    ]
    sections += [
        Section(
            layer=layer,
            width=via_stack_width + 2 * cladding_offset,
            offset=-via_stack_offset,
        )
        for layer, cladding_offset in zip(layers_via_stack2, bbox_offsets_via_stack2)
    ]
    if layer_via and via_width and via_offsets:
        sections += [
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
        sections=tuple(sections),
        **kwargs,
    )


@xsection
def pn(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    layer_slab: LayerSpec = "SLAB90",
    gap_low_doping: float = 0.0,
    gap_medium_doping: float | None = 0.5,
    gap_high_doping: float | None = 1.0,
    offset_low_doping: float | None = 0.0,
    width_doping: float = 8.0,
    width_slab: float = 7.0,
    layer_p: LayerSpec | None = "P",
    layer_pp: LayerSpec | None = "PP",
    layer_ppp: LayerSpec | None = "PPP",
    layer_n: LayerSpec | None = "N",
    layer_np: LayerSpec | None = "NP",
    layer_npp: LayerSpec | None = "NPP",
    layer_via: LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: LayerSpec | None = None,
    width_metal: float = 1.0,
    port_names: tuple[str, str] = ("o1", "o2"),
    sections: tuple[Section, ...] | None = None,
    cladding_layers: "LayerSpecs | None" = None,
    cladding_offsets: "Floats | None" = None,
    cladding_simplify: "Floats | None" = None,
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
    slab_insets = (slab_inset,) * 2 if slab_inset else None

    slab = Section(width=width_slab, offset=0, layer=layer_slab, insets=slab_insets)

    sections = list(sections or [])
    sections += [slab]
    base_offset_low_doping = width_doping / 2 + gap_low_doping / 4
    width_low_doping = width_doping - gap_low_doping / 2

    if layer_n:
        n = Section(
            width=width_low_doping + offset_low_doping,
            offset=+base_offset_low_doping - offset_low_doping / 2,
            layer=layer_n,
        )
        sections.append(n)
    if layer_p:
        p = Section(
            width=width_low_doping - offset_low_doping,
            offset=-base_offset_low_doping - offset_low_doping / 2,
            layer=layer_p,
        )
        sections.append(p)

    if gap_medium_doping is not None:
        width_medium_doping = width_doping - gap_medium_doping
        offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

        if layer_np is not None:
            np = Section(
                width=width_medium_doping,
                offset=+offset_medium_doping,
                layer=layer_np,
            )
            sections.append(np)
        if layer_pp is not None:
            pp = Section(
                width=width_medium_doping,
                offset=-offset_medium_doping,
                layer=layer_pp,
            )
            sections.append(pp)

    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width_high_doping / 2 + gap_high_doping
        if layer_npp is not None:
            npp = Section(
                width=width_high_doping, offset=+offset_high_doping, layer=layer_npp
            )
            sections.append(npp)
        if layer_ppp is not None:
            ppp = Section(
                width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp
            )
            sections.append(ppp)

    if layer_via is not None:
        offset = width_high_doping + gap_high_doping - width_via / 2
        via_top = Section(width=width_via, offset=+offset, layer=layer_via)
        via_bot = Section(width=width_via, offset=-offset, layer=layer_via)
        sections.append(via_top)
        sections.append(via_bot)

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
        sections.append(metal_top)
        sections.append(metal_bot)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=tuple(sections),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        cladding_simplify=cladding_simplify,
        **kwargs,
    )


@xsection
def pn_with_trenches(
    width: float = 0.5,
    layer: LayerSpec | None = None,
    layer_trench: LayerSpec = "DEEP_ETCH",
    gap_low_doping: float = 0.0,
    gap_medium_doping: float | None = 0.5,
    gap_high_doping: float | None = 1.0,
    offset_low_doping: float | None = 0.0,
    width_doping: float = 8.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    width_trench: float = 2.0,
    layer_p: LayerSpec | None = "P",
    layer_pp: LayerSpec | None = "PP",
    layer_ppp: LayerSpec | None = "PPP",
    layer_n: LayerSpec | None = "N",
    layer_np: LayerSpec | None = "NP",
    layer_npp: LayerSpec | None = "NPP",
    layer_via: LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: LayerSpec | None = None,
    width_metal: float = 1.0,
    port_names: "PortNames" = ("o1", "o2"),
    cladding_layers: Layers | None = cladding_layers_optical,
    cladding_offsets: "Floats | None" = cladding_offsets_optical,
    cladding_simplify: "Floats | None" = cladding_simplify_optical,
    wg_marking_layer: LayerSpec | None = None,
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
    sections = []
    sections += list(sections or [])
    sections += [Section(width=width_slab, layer=layer)]
    sections += [
        Section(width=width_trench, offset=offset, layer=layer_trench)
        for offset in [+trench_offset, -trench_offset]
    ]

    if wg_marking_layer is not None:
        sections += [Section(width=width, offset=0, layer=wg_marking_layer)]

    base_offset_low_doping = width_doping / 2 + gap_low_doping / 4
    width_low_doping = width_doping - gap_low_doping / 2

    if layer_n:
        n = Section(
            width=width_low_doping + offset_low_doping,
            offset=+base_offset_low_doping - offset_low_doping / 2,
            layer=layer_n,
        )
        sections.append(n)
    if layer_p:
        p = Section(
            width=width_low_doping - offset_low_doping,
            offset=-base_offset_low_doping - offset_low_doping / 2,
            layer=layer_p,
        )
        sections.append(p)

    if gap_medium_doping is not None:
        width_medium_doping = width_doping - gap_medium_doping
        offset_medium_doping = width_medium_doping / 2 + gap_medium_doping

        if layer_np:
            np = Section(
                width=width_medium_doping,
                offset=+offset_medium_doping,
                layer=layer_np,
            )
            sections.append(np)
        if layer_pp:
            pp = Section(
                width=width_medium_doping,
                offset=-offset_medium_doping,
                layer=layer_pp,
            )
            sections.append(pp)

    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width_high_doping / 2 + gap_high_doping
        if layer_npp:
            npp = Section(
                width=width_high_doping, offset=+offset_high_doping, layer=layer_npp
            )
            sections.append(npp)
        if layer_ppp:
            ppp = Section(
                width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp
            )
            sections.append(ppp)

    if layer_via is not None:
        offset = width_high_doping + gap_high_doping - width_via / 2
        via_top = Section(width=width_via, offset=+offset, layer=layer_via)
        via_bot = Section(width=width_via, offset=-offset, layer=layer_via)
        sections.append(via_top)
        sections.append(via_bot)

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
        sections.append(metal_top)
        sections.append(metal_bot)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=tuple(sections),
        cladding_offsets=cladding_offsets,
        cladding_simplify=cladding_simplify,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def pn_with_trenches_asymmetric(
    width: float = 0.5,
    layer: LayerSpec | None = None,
    layer_trench: LayerSpec = "DEEP_ETCH",
    gap_low_doping: float | tuple[float, float] = (0.0, 0.0),
    gap_medium_doping: float | tuple[float, float] | None = (0.5, 0.2),
    gap_high_doping: float | tuple[float, float] | None = (1.0, 0.8),
    width_doping: float = 8.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    width_trench: float = 2.0,
    layer_p: LayerSpec | None = "P",
    layer_pp: LayerSpec | None = "PP",
    layer_ppp: LayerSpec | None = "PPP",
    layer_n: LayerSpec | None = "N",
    layer_np: LayerSpec | None = "NP",
    layer_npp: LayerSpec | None = "NPP",
    layer_via: LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: LayerSpec | None = None,
    width_metal: float = 1.0,
    port_names: tuple[str, str] = ("o1", "o2"),
    cladding_layers: Layers | None = cladding_layers_optical,
    cladding_offsets: "Floats | None" = cladding_offsets_optical,
    wg_marking_layer: LayerSpec | None = None,
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
    sections = []
    sections += list(sections or [])
    sections += [Section(width=width_slab, layer=layer)]
    sections += [
        Section(width=width_trench, offset=offset, layer=layer_trench)
        for offset in [+trench_offset, -trench_offset]
    ]

    if wg_marking_layer is not None:
        sections += [Section(width=width, offset=0, layer=wg_marking_layer)]

    # Low doping
    if not isinstance(gap_low_doping, list | tuple):
        gap_low_doping = [gap_low_doping] * 2

    if layer_n:
        width_low_doping_n = width_doping - gap_low_doping[1]
        n = Section(
            width=width_low_doping_n,
            offset=width_low_doping_n / 2 + gap_low_doping[1],
            layer=layer_n,
        )
        sections.append(n)
    if layer_p:
        width_low_doping_p = width_doping - gap_low_doping[0]
        p = Section(
            width=width_low_doping_p,
            offset=-(width_low_doping_p / 2 + gap_low_doping[0]),
            layer=layer_p,
        )
        sections.append(p)

    if gap_medium_doping is not None:
        if not isinstance(gap_medium_doping, list | tuple):
            gap_medium_doping = [gap_medium_doping] * 2

        if layer_np:
            width_np = width_doping - gap_medium_doping[1]
            np = Section(
                width=width_np,
                offset=width_np / 2 + gap_medium_doping[1],
                layer=layer_np,
            )
            sections.append(np)
        if layer_pp:
            width_pp = width_doping - gap_medium_doping[0]
            pp = Section(
                width=width_pp,
                offset=-(width_pp / 2 + gap_medium_doping[0]),
                layer=layer_pp,
            )
            sections.append(pp)

    if gap_high_doping is not None:
        if not isinstance(gap_high_doping, list | tuple):
            gap_high_doping = [gap_high_doping] * 2

        if layer_npp:
            width_npp = width_doping - gap_high_doping[1]
            npp = Section(
                width=width_npp,
                offset=width_npp / 2 + gap_high_doping[1],
                layer=layer_npp,
            )
            sections.append(npp)
        if layer_ppp:
            width_ppp = width_doping - gap_high_doping[0]
            ppp = Section(
                width=width_ppp,
                offset=-(width_ppp / 2 + gap_high_doping[0]),
                layer=layer_ppp,
            )
            sections.append(ppp)

    if layer_via is not None:
        offset_top = width_npp + gap_high_doping[1] - width_via / 2
        offset_bot = width_ppp + gap_high_doping[0] - width_via / 2
        via_top = Section(width=width_via, offset=+offset_top, layer=layer_via)
        via_bot = Section(width=width_via, offset=-offset_bot, layer=layer_via)
        sections.append(via_top)
        sections.append(via_bot)

    if layer_metal is not None:
        offset_top = width_npp + gap_high_doping[1] - width_metal / 2
        offset_bot = width_ppp + gap_high_doping[0] - width_metal / 2
        port_types = ("electrical", "electrical")
        metal_top = Section(
            width=width_via,
            offset=+offset_top,
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
        sections.append(metal_top)
        sections.append(metal_bot)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=tuple(sections),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def l_wg_doped_with_trenches(
    width: float = 0.5,
    layer: LayerSpec | None = None,
    layer_trench: LayerSpec = "DEEP_ETCH",
    gap_low_doping: float = 0.0,
    gap_medium_doping: float | None = 0.5,
    gap_high_doping: float | None = 1.0,
    width_doping: float = 8.0,
    slab_offset: float | None = 0.3,
    width_slab: float | None = None,
    width_trench: float = 2.0,
    layer_low: LayerSpec = "P",
    layer_mid: LayerSpec = "PP",
    layer_high: LayerSpec = "PPP",
    layer_via: LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: LayerSpec | None = None,
    width_metal: float = 1.0,
    port_names: tuple[str, str] = ("o1", "o2"),
    cladding_layers: Layers | None = cladding_layers_optical,
    cladding_offsets: "Floats | None" = cladding_offsets_optical,
    wg_marking_layer: LayerSpec | None = None,
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
    sections = []
    sections += list(sections or [])
    sections += [
        Section(width=width_slab, layer=layer, offset=-1 * (width_slab / 2 - width / 2))
    ]
    sections += [Section(width=width_trench, offset=trench_offset, layer=layer_trench)]

    if wg_marking_layer is not None:
        sections += [Section(width=width, offset=0, layer=wg_marking_layer)]

    offset_low_doping = width / 2 - gap_low_doping - width_doping / 2

    low_doping = Section(
        width=width_doping,
        offset=offset_low_doping,
        layer=layer_low,
    )

    sections.append(low_doping)

    if gap_medium_doping is not None:
        width_medium_doping = width_doping - gap_medium_doping
        offset_medium_doping = width / 2 - gap_medium_doping - width_medium_doping / 2

        mid_doping = Section(
            width=width_medium_doping,
            offset=offset_medium_doping,
            layer=layer_mid,
        )
        sections.append(mid_doping)

    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width / 2 - gap_high_doping - width_high_doping / 2

        high_doping = Section(
            width=width_high_doping, offset=+offset_high_doping, layer=layer_high
        )

        sections.append(high_doping)

    if layer_via is not None:
        offset = offset_high_doping - width_high_doping / 2 + width_via / 2
        via = Section(width=width_via, offset=+offset, layer=layer_via)
        sections.append(via)

    if layer_metal is not None:
        offset = offset_high_doping - width_high_doping / 2 + width_metal / 2
        port_types = ("electrical", "electrical")
        metal = Section(
            width=width_via,
            offset=+offset,
            layer=layer_metal,
            port_types=port_types,
            port_names=("e1_top", "e2_top"),
        )
        sections.append(metal)

    return cross_section(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=tuple(sections),
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        **kwargs,
    )


@xsection
def strip_heater_metal_undercut(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    heater_width: float = 2.5,
    trench_width: float = 6.5,
    trench_gap: float = 2.0,
    layer_heater: LayerSpec = "HEATER",
    layer_trench: LayerSpec = "DEEPTRENCH",
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
    sections = list(sections or [])
    sections += [
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
        sections=tuple(sections),
        **kwargs,
    )


@xsection
def strip_heater_metal(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    heater_width: float = 2.5,
    layer_heater: LayerSpec = "HEATER",
    sections: Sections | None = None,
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
        kwargs: cross_section settings.

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.strip_heater_metal(width=0.5, heater_width=2)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    sections = list(sections or [])
    sections += [
        Section(
            layer=layer_heater,
            width=heater_width,
            port_names=port_names_electrical,
            port_types=port_types_electrical,
        )
    ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(sections),
        **kwargs,
    )


@xsection
def strip_heater_doped(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    layers_heater: LayerSpecs = ("WG", "NPP"),
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

    sections = list(sections or [])
    sections += [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=+heater_offset,
            name=f"heater_upper_{layer}",
        )
        for layer, cladding_offset in zip(layers_heater, bbox_offsets_heater)
    ]

    sections += [
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
        sections=tuple(sections),
        **kwargs,
    )


@xsection
def rib_heater_doped(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    layer_heater: LayerSpec = "NPP",
    layer_slab: LayerSpec = "SLAB90",
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
        slab_offset = 0
    elif with_top_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = -slab_width / 2
    elif with_bot_heater:
        slab_width = width + heater_gap + heater_width + slab_gap
        slab_offset = +slab_width / 2

    sections = list(sections or [])
    sections += [
        Section(width=slab_width, layer=layer_slab, offset=slab_offset, name="slab")
    ]

    if with_bot_heater:
        sections += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=+heater_offset,
                name="heater_upper",
            )
        ]
    if with_top_heater:
        sections += [
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
        sections=tuple(sections),
        **kwargs,
    )


@xsection
def rib_heater_doped_via_stack(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    heater_width: float = 1.0,
    heater_gap: float = 0.8,
    layer_slab: LayerSpec = "SLAB90",
    layer_heater: LayerSpec = "NPP",
    via_stack_width: float = 2.0,
    via_stack_gap: float = 0.8,
    layers_via_stack: LayerSpecs = ("NPP", "VIAC"),
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

    heater_offset = width / 2 + heater_gap + heater_width / 2
    via_stack_offset = width / 2 + via_stack_gap + via_stack_width / 2
    sections = list(sections or [])
    sections += [
        Section(width=slab_width, layer=layer_slab, offset=slab_offset, name="slab"),
    ]
    if with_bot_heater:
        sections += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=+heater_offset,
            )
        ]

    if with_top_heater:
        sections += [
            Section(
                layer=layer_heater,
                width=heater_width,
                offset=-heater_offset,
            )
        ]

    if with_bot_heater:
        sections += [
            Section(
                layer=layer,
                width=heater_width + 2 * cladding_offset,
                offset=+via_stack_offset,
            )
            for layer, cladding_offset in zip(layers_via_stack, bbox_offsets_via_stack)
        ]

    if with_top_heater:
        sections += [
            Section(
                layer=layer,
                width=heater_width + 2 * cladding_offset,
                offset=-via_stack_offset,
            )
            for layer, cladding_offset in zip(layers_via_stack, bbox_offsets_via_stack)
        ]

    return strip(
        sections=tuple(sections),
        width=width,
        layer=layer,
        **kwargs,
    )


@xsection
def pn_ge_detector_si_contacts(
    width_si: float = 6.0,
    layer_si: LayerSpec = "WG",
    width_ge: float = 3.0,
    layer_ge: LayerSpec = "GE",
    gap_low_doping: float = 0.6,
    gap_medium_doping: float | None = 0.9,
    gap_high_doping: float | None = 1.1,
    width_doping: float = 8.0,
    layer_p: LayerSpec = "P",
    layer_pp: LayerSpec = "PP",
    layer_ppp: LayerSpec = "PPP",
    layer_n: LayerSpec = "N",
    layer_np: LayerSpec = "NP",
    layer_npp: LayerSpec = "NPP",
    layer_via: LayerSpec | None = None,
    width_via: float = 1.0,
    layer_metal: LayerSpec | None = None,
    port_names: tuple[str, str] = ("o1", "o2"),
    cladding_layers: Layers | None = cladding_layers_optical,
    cladding_offsets: "Floats | None" = cladding_offsets_optical,
    cladding_simplify: "Floats | None" = None,
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

    sections = [s, n, p]

    cladding_layers = cladding_layers or ()
    cladding_offsets = cladding_offsets or ()
    cladding_simplify = cladding_simplify or (None,) * len(cladding_layers)
    sections += [
        Section(width=width_si + 2 * offset, layer=layer, simplify=simplify)
        for layer, offset, simplify in zip(
            cladding_layers, cladding_offsets, cladding_simplify
        )
    ]

    if gap_medium_doping is not None:
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
        sections.extend((np, pp))
    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width_high_doping / 2 + gap_high_doping
        npp = Section(
            width=width_high_doping, offset=+offset_high_doping, layer=layer_npp
        )
        ppp = Section(
            width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp
        )
        sections.extend((npp, ppp))
    if layer_via is not None:
        offset = width_high_doping / 2 + gap_high_doping
        via_top = Section(width=width_via, offset=+offset, layer=layer_via)
        via_bot = Section(width=width_via, offset=-offset, layer=layer_via)
        sections.extend((via_top, via_bot))
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
        sections.extend((metal_top, metal_bot))

    # Add the Ge
    s = Section(width=width_ge, offset=0, layer=layer_ge)
    sections.append(s)

    return CrossSection(
        sections=tuple(sections),
        **kwargs,
    )


def get_cross_sections(
    modules: Iterable[ModuleType] | ModuleType, verbose: bool = False
) -> dict[str, CrossSectionFactory]:
    """Returns cross_sections from a module or list of modules.

    Args:
        modules: module or iterable of modules.
        verbose: prints in case any errors occur.
    """
    modules = modules if isinstance(modules, Iterable) else [modules]

    xs = {}
    for module in modules:
        for t in getmembers(module):
            if callable(t[1]) and not t[0].startswith("_"):
                try:
                    r = signature(
                        t[1].func if isinstance(t[1], partial) else t[1]
                    ).return_annotation
                    if r == CrossSection or (
                        isinstance(r, str) and r.endswith("CrossSection")
                    ):
                        xs[t[0]] = t[1]
                except Exception as e:
                    if verbose:
                        logger.warn(f"error in {t[0]}: {e}")
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
