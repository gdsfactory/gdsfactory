"""You can define a path as list of points.

To create a component you need to extrude the path with a cross-section.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
import functools
from collections.abc import Iterable
from functools import partial
from inspect import getmembers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar

from pydantic import BaseModel, Field, validate_arguments
from typing_extensions import Literal


Layer = Tuple[int, int]
Layers = Tuple[Layer, ...]
WidthTypes = Literal["sine", "linear", "parabolic"]

LayerSpec = Union[Layer, int, str]
LayerSpecs = Union[List[LayerSpec], Tuple[LayerSpec, ...]]
Floats = Tuple[float, ...]
port_names_electrical = ("e1", "e2")
port_types_electrical = ("electrical", "electrical")

cladding_layers_optical = None
cladding_offsets_optical = None


class Section(BaseModel):
    """CrossSection to extrude a path with a waveguide.

    Parameters:
        width: of the section (um) or parameterized function from 0 to 1.
             the width at t==0 is the width at the beginning of the Path.
             the width at t==1 is the width at the end.
        offset: center offset (um) or function parameterized function from 0 to 1.
             the offset at t==0 is the offset at the beginning of the Path.
             the offset at t==1 is the offset at the end.
        layer: layer spec. If None does not draw the main section.
        port_names: Optional port names.
        port_types: optical, electrical, ...
        name: Optional Section name.
        hidden: hide layer.

    .. code::

          0   offset
          |<-------------->|
          |              _____
          |             |     |
          |             |layer|
          |             |_____|
          |              <---->
                         width
    """

    width: Union[float, Callable]
    offset: Union[float, Callable] = 0
    layer: Optional[LayerSpec] = None
    port_names: Tuple[Optional[str], Optional[str]] = (None, None)
    port_types: Tuple[str, str] = ("optical", "optical")
    name: Optional[str] = None
    hidden: bool = False

    class Config:
        """pydantic basemodel config."""

        extra = "forbid"


class CrossSection(BaseModel):
    """Waveguide information to extrude a path.

    cladding_layers follow path shape, while bbox_layers are rectangular.

    Parameters:
        layer: main Section layer. Main section name = '_default'.
            If None does not draw the main section.
        width: main Section width (um) or function parameterized from 0 to 1.
            the width at t==0 is the width at the beginning of the Path.
            the width at t==1 is the width at the end.
        offset: main Section center offset (um) or function from 0 to 1.
             the offset at t==0 is the offset at the beginning of the Path.
             the offset at t==1 is the offset at the end.
        radius: main Section bend radius (um).
        width_wide: wide waveguides width (um) for low loss routing.
        auto_widen: taper to wide waveguides for low loss routing.
        auto_widen_minimum_length: minimum straight length for auto_widen.
        taper_length: taper_length for auto_widen.
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        cladding_layers: list of layers to extrude.
        cladding_offsets: list of offset from main Section edge.
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        gap: edge to edge waveguide gap for routing.
        min_length: defaults to 1nm = 10e-3um for routing.
        start_straight_length: straight length at the beginning of the route.
        end_straight_length: end length at the beginning of the route.
        snap_to_grid: Optional snap points to grid when extruding paths (um).
        decorator: function when extruding component. For example add_pins.
        add_pins: Optional function to add pins.
        add_bbox: Optional function to add bounding box.
        info: dict with extra settings or useful information.
        name: cross_section name.
        mirror: if True, reflects the offsets.

    Properties:
        aliases: dict of cross_section aliases.
    """

    layer: Optional[LayerSpec] = None
    width: Union[float, Callable]
    offset: Union[float, Callable] = 0
    radius: Optional[float] = None
    width_wide: Optional[float] = None
    auto_widen: bool = False
    auto_widen_minimum_length: float = 200.0
    taper_length: float = 10.0
    bbox_layers: List[LayerSpec] = Field(default_factory=list)
    bbox_offsets: List[float] = Field(default_factory=list)
    cladding_layers: Optional[LayerSpecs] = None
    cladding_offsets: Optional[Floats] = None
    sections: List[Section] = Field(default_factory=list)
    port_names: Tuple[str, str] = ("o1", "o2")
    port_types: Tuple[str, str] = ("optical", "optical")
    gap: float = 3.0
    min_length: float = 10e-3
    start_straight_length: float = 10e-3
    end_straight_length: float = 10e-3
    snap_to_grid: Optional[float] = None
    decorator: Optional[Callable] = None
    add_pins: Optional[Callable] = None
    add_bbox: Optional[Callable] = None
    info: Dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None
    mirror: bool = False

    def __init__(__pydantic_self__, **data: Any) -> None:
        """Extend BaseModel init to process mirroring."""
        super().__init__(**data)

        if "mirror" in data and data["mirror"]:
            data["offset"] *= -1
            for section in data["sections"]:
                section.offset *= -1
            for offset in data["cladding_offsets"]:
                offset *= -1

    class Config:
        """Configuration."""

        extra = "forbid"
        fields = {
            "decorator": {"exclude": True},
            "add_pins": {"exclude": True},
            "add_bbox": {"exclude": True},
        }

    def copy(self, **kwargs):
        """Returns a CrossSection copy."""
        xs = super().copy(update=kwargs)
        xs.decorator = self.decorator
        xs.add_pins = self.add_pins
        xs.add_bbox = self.add_bbox
        return xs

    def get_name(self) -> str:
        h = hashlib.md5(str(self).encode()).hexdigest()[:8]
        return f"xs_{h}"

    @property
    def aliases(self) -> Dict[str, Section]:
        s = {
            "_default": Section(
                width=self.width,
                offset=self.offset,
                layer=self.layer,
                port_names=self.port_names,
                port_types=self.port_types,
                name="_default",
            )
        }
        sections = self.sections or []
        for section in sections:
            if section.name:
                s[section.name] = section
        return s

    def add_bbox_layers(
        self,
        component,
        top: Optional[float] = None,
        bottom: Optional[float] = None,
        right: Optional[float] = None,
        left: Optional[float] = None,
    ):
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
        x = self
        if x.bbox_layers and x.bbox_offsets:
            padding = []
            for offset in x.bbox_offsets:
                points = get_padding_points(
                    component=c,
                    default=0,
                    top=top or offset,
                    bottom=bottom or offset,
                    left=left or offset,
                    right=right or offset,
                )
                padding.append(points)

            for layer, points in zip(x.bbox_layers, padding):
                c.add_polygon(points, layer=layer)
        return c


class Transition(BaseModel):
    """Waveguide information to extrude a path between two CrossSection.

    cladding_layers follow path shape, while bbox_layers are rectangular.

    Parameters:
        cross_section1: input cross_section.
        cross_section2: output cross_section.
        width_type: sine or linear.
          Sets the type of width transition used if any widths are different
          between the two input CrossSections.
        sections: list of Sections(width, offset, layer, ports).
        layer: main Section layer. Main section name = '_default'.
        width: main Section width (um) or function parameterized from 0 to 1.
            the width at t==0 is the width at the beginning of the Path.
            the width at t==1 is the width at the end.
        snap_to_grid: Optional snap points to grid when extruding paths (um).
    """

    cross_section1: CrossSection
    cross_section2: CrossSection
    width_type: WidthTypes = "sine"
    sections: List[Section]
    layer: Optional[LayerSpec] = None
    width: Optional[Union[float, Callable]] = None
    snap_to_grid: Optional[float] = None


def _xsection_without_validator(func):
    """Decorator for cross_section functions

    use xsection instead so it will validate arguments with types.
    """

    @functools.wraps(func)
    def _xsection(*args, **kwargs):
        xs = func(*args, **kwargs)

        sig = inspect.signature(func)
        args_as_kwargs = dict(zip(sig.parameters.keys(), args))
        args_as_kwargs.update(kwargs)

        if not isinstance(xs, CrossSection):
            raise ValueError(
                f"function {func.__name__!r} return type = {type(xs)}",
                "make sure that functions with @xsection decorator return a CrossSection",
            )

        xs.info.update(settings=args_as_kwargs, function_name=func.__name__)
        return xs

    return _xsection


_F = TypeVar("_F", bound=Callable)


def xsection(func: _F) -> _F:
    """Decorator for CrossSection functions

    Validates type annotations with pydantic.

    .. plot::
        :include-source:

        import gdsfactory as gf

        @gf.cross_section.xsection
        def xs_sc(width=0.5, **kwargs):
            xs = gf.cross_section.cross_section(width=width, **kwargs)
            return xs

        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs_sc)
        c.plot()
    """
    return _xsection_without_validator(validate_arguments(func))


@xsection
def cross_section(
    width: Union[Callable, float] = 0.5,
    offset: Union[float, Callable] = 0,
    layer: Optional[LayerSpec] = "WG",
    width_wide: Optional[float] = None,
    auto_widen: bool = False,
    auto_widen_minimum_length: float = 200.0,
    taper_length: float = 10.0,
    radius: Optional[float] = 10.0,
    sections: Optional[Tuple[Section, ...]] = None,
    port_names: Tuple[str, str] = ("o1", "o2"),
    port_types: Tuple[str, str] = ("optical", "optical"),
    gap: float = 3.0,
    min_length: float = 10e-3,
    start_straight_length: float = 10e-3,
    end_straight_length: float = 10e-3,
    snap_to_grid: Optional[float] = None,
    bbox_layers: Optional[List[LayerSpec]] = None,
    bbox_offsets: Optional[List[float]] = None,
    cladding_layers: Optional[LayerSpecs] = None,
    cladding_offsets: Optional[Floats] = None,
    info: Optional[Dict[str, Any]] = None,
    decorator: Optional[Callable] = None,
    add_pins: Optional[Callable] = None,
    add_bbox: Optional[Callable] = None,
    mirror: bool = False,
    name: Optional[str] = None,
) -> CrossSection:
    """Return CrossSection.

    Args:
        width: main Section width (um) or function parameterized from 0 to 1.
            the width at t==0 is the width at the beginning of the Path.
            the width at t==1 is the width at the end.
        offset: main Section center offset (um) or function from 0 to 1.
             the offset at t==0 is the offset at the beginning of the Path.
             the offset at t==1 is the offset at the end.
        layer: main section layer.
        width_wide: wide waveguides width (um) for low loss routing.
        auto_widen: taper to wide waveguides for low loss routing.
        auto_widen_minimum_length: minimum straight length for auto_widen.
        taper_length: taper_length for auto_widen.
        radius: bend radius (um).
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        gap: edge to edge waveguide gap for routing.
        min_length: defaults to 1nm = 10e-3um for routing.
        start_straight_length: straight length at the beginning of the route.
        end_straight_length: end length at the beginning of the route.
        snap_to_grid: can snap points to grid when extruding the path.
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        cladding_layers: list of layers to extrude.
        cladding_offsets: list of offset from main Section edge.
        info: settings info.
        decorator: function to run when converting path to component.
        add_pins: optional function to add pins to component.
        add_bbox: optional function to add bounding box to component.
        mirror: if True, reflects the offsets.
        name: cross_section name.


    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.cross_section(width=0.5, offset=0, layer='WG')
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    return CrossSection(
        width=width,
        offset=offset,
        layer=layer,
        width_wide=width_wide,
        auto_widen=auto_widen,
        auto_widen_minimum_length=auto_widen_minimum_length,
        taper_length=taper_length,
        radius=radius,
        bbox_layers=bbox_layers or [],
        bbox_offsets=bbox_offsets or [],
        cladding_layers=cladding_layers,
        cladding_offsets=cladding_offsets,
        sections=sections or (),
        gap=gap,
        min_length=min_length,
        start_straight_length=start_straight_length,
        end_straight_length=end_straight_length,
        snap_to_grid=snap_to_grid,
        port_types=port_types,
        port_names=port_names,
        info=info or {},
        decorator=decorator,
        add_bbox=add_bbox,
        add_pins=add_pins,
        mirror=mirror,
        name=name,
    )


strip = cross_section
strip_auto_widen = partial(strip, width_wide=0.9, auto_widen=True)
strip_no_pins = partial(
    strip, add_pins=None, add_bbox=None, cladding_layers=None, cladding_offsets=None
)

# Rib with rectangular slab
rib = partial(
    strip,
    bbox_layers=["SLAB90"],
    bbox_offsets=[3],
)

# Rib with with slab that follows the waveguide core
rib_conformal = partial(
    strip,
    sections=(Section(width=6, layer="SLAB90", name="slab"),),
)
nitride = partial(strip, layer="WGN", width=1.0)
strip_rib_tip = partial(
    strip, sections=(Section(width=0.2, layer="SLAB90", name="slab"),)
)


@xsection
def slot(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    slot_width: float = 0.04,
    **kwargs,
) -> CrossSection:
    """Return CrossSection Slot (with an etched region in the center).

    Args:
        width: main Section width (um) or function parameterized from 0 to 1.
            the width at t==0 is the width at the beginning of the Path.
            the width at t==1 is the width at the end.
        layer: main section layer.
        slot_width: in um.

    Keyword Args:
        offset: main Section center offset (um) or function from 0 to 1.
             the offset at t==0 is the offset at the beginning of the Path.
             the offset at t==1 is the offset at the end.
        width_wide: wide waveguides width (um) for low loss routing.
        auto_widen: taper to wide waveguides for low loss routing.
        auto_widen_minimum_length: minimum straight length for auto_widen.
        taper_length: taper_length for auto_widen.
        radius: bend radius (um).
        sections: list of Sections(width, offset, layer, ports).
        port_names: for input and output ('o1', 'o2').
        port_types: for input and output: electrical, optical, vertical_te ...
        min_length: defaults to 1nm = 10e-3um for routing.
        start_straight_length: straight length at the beginning of the route.
        end_straight_length: end length at the beginning of the route.
        snap_to_grid: can snap points to grid when extruding the path.
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        cladding_layers: list of layers to extrude.
        cladding_offsets: list of offset from main Section edge.
        info: settings info.
        decorator: function to run when converting path to component.
        add_pins: optional function to add pins to component.
        add_bbox: optional function to add bounding box to component.

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
    sections = [
        Section(width=rail_width, offset=rail_offset, layer=layer, name="left_rail"),
        Section(width=rail_width, offset=-rail_offset, layer=layer, name="right rail"),
    ]

    return strip(
        width=width,
        layer=None,
        sections=tuple(sections),
        **kwargs,
    )


@xsection
def rib_with_trenches(
    width: float = 0.5,
    width_trench: float = 2.0,
    width_slab: float = 7.0,
    layer: Optional[LayerSpec] = "WG",
    layer_trench: LayerSpec = "DEEP_ETCH",
    **kwargs,
) -> CrossSection:
    """Return CrossSection of rib waveguide defined by trenches.

    Args:
        width: main Section width (um) or function parameterized from 0 to 1.
            the width at t==0 is the width at the beginning of the Path.
            the width at t==1 is the width at the end.
        width_slab: in um.
        width_trench: in um.
        layer: ridge layer. None adds only ridge.
        layer_trench: layer to etch trenches.
        kwargs: cross_section settings.


    .. code::


        _____         __________         ________
             |        |         |        |
             |________|         |________|

       __________________________________________
             <------->                           |
            width_trench
                                                 |
       <---------------------------------------->



    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.rib_with_trenches(width=0.5)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    width_slab = max(width_slab, width + 2 * width_trench)

    trench_offset = width / 2 + width_trench / 2
    sections = [Section(width=width_slab, layer=layer)]
    sections += [
        Section(width=width_trench, offset=offset, layer=layer_trench)
        for offset in [+trench_offset, -trench_offset]
    ]

    return CrossSection(
        width=width,
        layer=None,
        sections=tuple(sections),
        **kwargs,
    )


metal1 = partial(
    cross_section,
    layer="M1",
    width=10.0,
    port_names=port_names_electrical,
    port_types=port_types_electrical,
    radius=None,
)
metal2 = partial(
    metal1,
    layer="M2",
)
metal3 = partial(
    metal1,
    layer="M3",
)
heater_metal = partial(
    metal1,
    width=2.5,
    layer="HEATER",
)

metal3_with_bend = partial(metal1, layer="M3", radius=10)
metal_routing = metal3
npp = partial(metal1, layer="NPP", width=0.5)


@xsection
def pin(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    layer_slab: LayerSpec = "SLAB90",
    layers_via_stack1: LayerSpecs = ("PPP",),
    layers_via_stack2: LayerSpecs = ("NPP",),
    bbox_offsets_via_stack1: Tuple[float, ...] = (0, -0.2),
    bbox_offsets_via_stack2: Tuple[float, ...] = (0, -0.2),
    via_stack_width: float = 9.0,
    via_stack_gap: float = 0.55,
    slab_gap: float = -0.2,
    layer_via: LayerSpec = None,
    via_width: float = 1,
    via_offsets: Optional[Tuple[float, ...]] = None,
    **kwargs,
) -> CrossSection:
    """Rib PIN doped cross_section.

    Args:
        width: ridge width.
        layer: ridge layer.
        layer_slab: slab layer.
        layers_via_stack1: P++ layer.
        layers_via_stack2: N++ layer.
        bbox_offsets_via_stack1: for via left.
        bbox_offsets_via_stack2: for via right.
        via_stack_width: in um.
        via_stack_gap: offset from via_stack to ridge edge.
        slab_gap: extra slab gap (negative: via_stack goes beyond slab).
        layer_via: for via.
        via_width: in um.
        via_offsets: in um.
        kwargs: other cross_section settings.

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
    slab_width = width + 2 * via_stack_gap + 2 * via_stack_width - 2 * slab_gap
    via_stack_offset = width / 2 + via_stack_gap + via_stack_width / 2

    sections = [Section(width=slab_width, layer=layer_slab, name="slab")]
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
    gap_medium_doping: Optional[float] = 0.5,
    gap_high_doping: Optional[float] = 1.0,
    offset_low_doping: Optional[float] = 0.0,
    width_doping: float = 8.0,
    width_slab: float = 7.0,
    layer_p: LayerSpec = "P",
    layer_pp: LayerSpec = "PP",
    layer_ppp: LayerSpec = "PPP",
    layer_n: LayerSpec = "N",
    layer_np: LayerSpec = "NP",
    layer_npp: LayerSpec = "NPP",
    layer_via: Optional[LayerSpec] = None,
    width_via: float = 1.0,
    layer_metal: Optional[LayerSpec] = None,
    width_metal: float = 1.0,
    port_names: Tuple[str, str] = ("o1", "o2"),
    bbox_layers: Optional[List[Layer]] = None,
    bbox_offsets: Optional[List[float]] = None,
    cladding_layers: Optional[Layers] = cladding_layers_optical,
    cladding_offsets: Optional[Floats] = cladding_offsets_optical,
    mirror: bool = False,
) -> CrossSection:
    """Rib PN doped cross_section.

    Args:
        width: width of the ridge in um.
        layer: ridge layer.
        layer_slab: slab layer.
        gap_low_doping: from waveguide center to low doping. Only used for PIN.
        gap_medium_doping: from waveguide center to medium doping.
            None removes medium doping.
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
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        mirror: if True, reflects all doping sections.

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
    slab = Section(width=width_slab, offset=0, layer=layer_slab)
    sections = [slab]
    base_offset_low_doping = width_doping / 2 + gap_low_doping / 4
    width_low_doping = width_doping - gap_low_doping / 2

    n = Section(
        width=width_low_doping + offset_low_doping,
        offset=+base_offset_low_doping - offset_low_doping / 2,
        layer=layer_n,
    )
    p = Section(
        width=width_low_doping - offset_low_doping,
        offset=-base_offset_low_doping - offset_low_doping / 2,
        layer=layer_p,
    )
    sections.append(n)
    sections.append(p)

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
        sections.append(np)
        sections.append(pp)

    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width_high_doping / 2 + gap_high_doping
        npp = Section(
            width=width_high_doping, offset=+offset_high_doping, layer=layer_npp
        )
        ppp = Section(
            width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp
        )
        sections.append(npp)
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

    bbox_layers = bbox_layers or []
    bbox_offsets = bbox_offsets or []
    for layer_cladding, cladding_offset in zip(bbox_layers, bbox_offsets):
        s = Section(
            width=width_slab + 2 * cladding_offset, offset=0, layer=layer_cladding
        )
        sections.append(s)

    return CrossSection(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=sections,
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        mirror=mirror,
    )


@xsection
def pn_with_trenches(
    width: float = 0.5,
    layer: Optional[LayerSpec] = None,
    layer_trench: LayerSpec = "DEEP_ETCH",
    gap_low_doping: float = 0.0,
    gap_medium_doping: Optional[float] = 0.5,
    gap_high_doping: Optional[float] = 1.0,
    offset_low_doping: Optional[float] = 0.0,
    width_doping: float = 8.0,
    width_slab: float = 7.0,
    width_trench: float = 2.0,
    layer_p: LayerSpec = "P",
    layer_pp: LayerSpec = "PP",
    layer_ppp: LayerSpec = "PPP",
    layer_n: LayerSpec = "N",
    layer_np: LayerSpec = "NP",
    layer_npp: LayerSpec = "NPP",
    layer_via: Optional[LayerSpec] = None,
    width_via: float = 1.0,
    layer_metal: Optional[LayerSpec] = None,
    width_metal: float = 1.0,
    port_names: Tuple[str, str] = ("o1", "o2"),
    bbox_layers: Optional[List[Layer]] = None,
    bbox_offsets: Optional[List[float]] = None,
    cladding_layers: Optional[Layers] = cladding_layers_optical,
    cladding_offsets: Optional[Floats] = cladding_offsets_optical,
    mirror: bool = False,
    **kwargs,
) -> CrossSection:
    """Rib PN doped cross_section.

    Args:
        width: width of the ridge in um.
        layer: ridge layer. None adds only ridge.
        layer_trench: layer to etch trenches.
        gap_low_doping: from waveguide center to low doping. Only used for PIN.
        gap_medium_doping: from waveguide center to medium doping.
            None removes medium doping.
        gap_high_doping: from center to high doping. None removes it.
        offset_low_doping: from center to junction center.
        width_doping: in um.
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
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        cladding_layers: optional list of cladding layers.
        cladding_offsets: optional list of cladding offsets.
        mirror: if True, reflects all doping sections.
        kwargs: cross_section settings.

    .. code::

                                   offset_low_doping
                                     <------>
                                    |       |
                                   wg     junction
                                 center   center
                                    |       |
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
    trench_offset = width / 2 + width_trench / 2
    sections = [Section(width=width_slab, layer=layer)]
    sections += [
        Section(width=width_trench, offset=offset, layer=layer_trench)
        for offset in [+trench_offset, -trench_offset]
    ]
    base_offset_low_doping = width_doping / 2 + gap_low_doping / 4
    width_low_doping = width_doping - gap_low_doping / 2

    n = Section(
        width=width_low_doping + offset_low_doping,
        offset=+base_offset_low_doping - offset_low_doping / 2,
        layer=layer_n,
    )
    p = Section(
        width=width_low_doping - offset_low_doping,
        offset=-base_offset_low_doping - offset_low_doping / 2,
        layer=layer_p,
    )
    sections.append(n)
    sections.append(p)

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
        sections.append(np)
        sections.append(pp)

    if gap_high_doping is not None:
        width_high_doping = width_doping - gap_high_doping
        offset_high_doping = width_high_doping / 2 + gap_high_doping
        npp = Section(
            width=width_high_doping, offset=+offset_high_doping, layer=layer_npp
        )
        ppp = Section(
            width=width_high_doping, offset=-offset_high_doping, layer=layer_ppp
        )
        sections.append(npp)
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

    bbox_layers = bbox_layers or []
    bbox_offsets = bbox_offsets or []
    for layer_cladding, cladding_offset in zip(bbox_layers, bbox_offsets):
        s = Section(
            width=width_slab + 2 * cladding_offset, offset=0, layer=layer_cladding
        )
        sections.append(s)

    return CrossSection(
        width=width,
        offset=0,
        layer=layer,
        port_names=port_names,
        sections=sections,
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
        mirror=mirror,
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
    **kwargs,
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
    return strip(
        width=width,
        layer=layer,
        sections=(
            Section(
                layer=layer_heater,
                width=heater_width,
                port_names=port_names_electrical,
                port_types=port_types_electrical,
            ),
            Section(layer=layer_trench, width=trench_width, offset=+trench_offset),
            Section(layer=layer_trench, width=trench_width, offset=-trench_offset),
        ),
        **kwargs,
    )


@xsection
def strip_heater_metal(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    heater_width: float = 2.5,
    layer_heater: LayerSpec = "HEATER",
    **kwargs,
) -> CrossSection:
    """Returns strip cross_section with top heater metal.

    dimensions from https://doi.org/10.1364/OE.18.020298

    Args:
        width: waveguide width (um).
        layer: waveguide layer.
        heater_width: of metal heater.
        layer_heater: for the metal.

    .. plot::
        :include-source:

        import gdsfactory as gf

        xs = gf.cross_section.strip_heater_metal(width=0.5, heater_width=2)
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """

    return strip(
        width=width,
        layer=layer,
        sections=(
            Section(
                layer=layer_heater,
                width=heater_width,
                port_names=port_names_electrical,
                port_types=port_types_electrical,
            ),
        ),
        **kwargs,
    )


@xsection
def strip_heater_doped(
    width: float = 0.5,
    layer: LayerSpec = "WG",
    heater_width: float = 2.0,
    heater_gap: float = 0.8,
    layers_heater: LayerSpecs = ("WG", "NPP"),
    bbox_offsets_heater: Tuple[float, ...] = (0, 0.1),
    **kwargs,
) -> CrossSection:
    """Returns strip cross_section with N++ doped heaters on both sides.

    Args:
        width: in um.
        layer: waveguide spec.
        heater_width: in um.
        heater_gap: in um.
        layers_heater: for doped heater.
        bbox_offsets_heater: for each layers_heater.
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

    sections = [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=+heater_offset,
        )
        for layer, cladding_offset in zip(layers_heater, bbox_offsets_heater)
    ]

    sections += [
        Section(
            layer=layer,
            width=heater_width + 2 * cladding_offset,
            offset=-heater_offset,
        )
        for layer, cladding_offset in zip(layers_heater, bbox_offsets_heater)
    ]

    return strip(
        width=width,
        layer=layer,
        sections=tuple(sections),
        **kwargs,
    )


strip_heater_doped_via_stack = partial(
    strip_heater_doped,
    layers_heater=("WG", "NPP", "VIAC"),
    bbox_offsets_heater=(0, 0.1, -0.2),
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
    **kwargs,
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

    sections = []

    if with_bot_heater:
        sections += [
            Section(layer=layer_heater, width=heater_width, offset=+heater_offset)
        ]
    if with_top_heater:
        sections += [
            Section(layer=layer_heater, width=heater_width, offset=-heater_offset)
        ]
    sections += [
        Section(width=slab_width, layer=layer_slab, offset=slab_offset, name="slab")
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
    bbox_offsets_via_stack: Tuple[float, ...] = (0, -0.2),
    slab_gap: float = 0.2,
    slab_offset: float = 0,
    with_top_heater: bool = True,
    with_bot_heater: bool = True,
    **kwargs,
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
    sections = [
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
    gap_medium_doping: Optional[float] = 0.9,
    gap_high_doping: Optional[float] = 1.1,
    width_doping: float = 8.0,
    layer_p: LayerSpec = "P",
    layer_pp: LayerSpec = "PP",
    layer_ppp: LayerSpec = "PPP",
    layer_n: LayerSpec = "N",
    layer_np: LayerSpec = "NP",
    layer_npp: LayerSpec = "NPP",
    layer_via: LayerSpec = None,
    width_via: float = 1.0,
    layer_metal: LayerSpec = None,
    port_names: Tuple[str, str] = ("o1", "o2"),
    bbox_layers: Optional[List[Layer]] = None,
    bbox_offsets: Optional[List[float]] = None,
    cladding_layers: Optional[Layers] = cladding_layers_optical,
    cladding_offsets: Optional[Floats] = cladding_offsets_optical,
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
        gap_medium_doping: from waveguide center to medium doping.
            None removes medium doping.
        gap_high_doping: from center to high doping. None removes it.
        width_doping: distance from the waveguide center to the edge
            of the p (or n) dopings in um.
        layer_p: p doping layer.
        layer_pp: p+ doping layer.
        layer_ppp: p++ doping layer.
        layer_n: n doping layer.
        layer_np: n+ doping layer.
        layer_npp: n++ doping layer.
        layer_via: via layer.
        width_via: via width in um.
        layer_metal: metal layer.
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.
        port_names: for input and output ('o1', 'o2').
        bbox_layers: list of layers for rectangular bounding box.
        bbox_offsets: list of bounding box offsets.

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
        p = gf.path.arc(radius=10, angle=45)
        c = p.extrude(xs)
        c.plot()
    """
    width_low_doping = width_doping - gap_low_doping
    offset_low_doping = width_low_doping / 2 + gap_low_doping

    n = Section(width=width_low_doping, offset=+offset_low_doping, layer=layer_n)
    p = Section(width=width_low_doping, offset=-offset_low_doping, layer=layer_p)
    sections = [n, p]
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
    bbox_layers = bbox_layers or []
    bbox_offsets = bbox_offsets or []
    for layer_cladding, cladding_offset in zip(bbox_layers, bbox_offsets):
        s = Section(
            width=width_si + 2 * cladding_offset, offset=0, layer=layer_cladding
        )
        sections.append(s)

    # Add the Ge
    s = Section(width=width_ge, offset=0, layer=layer_ge)
    sections.append(s)

    return CrossSection(
        width=width_si,
        offset=0,
        layer=layer_si,
        port_names=port_names,
        sections=sections,
        cladding_offsets=cladding_offsets,
        cladding_layers=cladding_layers,
    )


CrossSectionFactory = Callable[..., CrossSection]


def get_cross_section_factories(
    modules, verbose: bool = False
) -> Dict[str, CrossSectionFactory]:
    """Returns cross_section factories from a module or list of modules.

    Args:
        modules: module or iterable of modules.
        verbose: prints in case any errors occur.
    """
    modules = modules if isinstance(modules, Iterable) else [modules]

    xs = {}
    for module in modules:
        for t in getmembers(module):
            if callable(t[1]) and t[0] != "partial":
                try:
                    r = inspect.signature(t[1]).return_annotation
                    if r == CrossSection or (
                        isinstance(r, str) and r.endswith("CrossSection")
                    ):
                        xs[t[0]] = t[1]
                except ValueError:
                    if verbose:
                        print(f"error in {t[0]}")
    return xs


cross_sections = get_cross_section_factories(sys.modules[__name__])


def test_copy():
    import gdsfactory as gf

    p = gf.path.straight()
    copied_cs = gf.cross_section.strip().copy()
    gf.path.extrude(p, cross_section=copied_cs)


if __name__ == "__main__":
    import gdsfactory as gf

    # xs = gf.cross_section.pin(
    #     width=0.5,
    #     # gap_low_doping=0.05,
    #     # width_doping=2.0,
    #     # offset_low_doping=0,
    #     mirror=False,
    # )
    # xs = pn_with_trenches(width=0.3)
    # xs = slot(width=0.3)
    xs = rib_with_trenches()
    p = gf.path.straight()
    c = p.extrude(xs)
    c.show(show_ports=True)
