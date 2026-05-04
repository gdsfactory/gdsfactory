"""Technology definitions."""

from collections.abc import Callable
from functools import partial, wraps
from typing import Any

import gdsfactory as gf
from doroutes.bundles import add_bundle_astar
from gdsfactory.cross_section import (
    CrossSection,
    cross_section,
    port_names_electrical,
    port_types_electrical,
)
from gdsfactory.technology import (
    LayerLevel,
    LayerMap,
    LayerStack,
    LogicalLayer,
)
from gdsfactory.typings import (
    Floats,
    Layer,
    LayerSpec,
    LayerSpecs,
)

from mypdk.config import PATH

nm = 1e-3


class LAYER(LayerMap):
    """LayerMap for technology."""

    WG: Layer = (1, 0)  # type: ignore
    SLAB: Layer = (5, 0)  # type: ignore
    FLOORPLAN: Layer = (99, 0)  # type: ignore
    HEATER: Layer = (39, 0)  # type: ignore
    GRA: Layer = (6, 0)  # type: ignore
    LBL: Layer = (100, 0)  # type: ignore
    PAD: Layer = (41, 0)  # type: ignore

    # labels for gdsfactory
    LABEL_SETTINGS: Layer = (100, 0)  # type: ignore
    LABEL_INSTANCE: Layer = (101, 0)  # type: ignore


def get_layer_stack(
    thickness_wg: float = 220 * nm,
    thickness_slab: float = 100 * nm,
    zmin_heater: float = 1.1,
    thickness_heater: float = 700 * nm,
    zmin_metal: float = 1.1,
    thickness_metal: float = 700 * nm,
) -> LayerStack:
    """Returns LayerStack.

    based on paper https://www.degruyter.com/document/doi/10.1515/nanoph-2013-0034/html

    Args:
        thickness_wg: waveguide thickness in um.
        thickness_slab: slab thickness in um.
        zmin_heater: TiN heater.
        thickness_heater: TiN thickness.
        zmin_metal: metal thickness in um.
        thickness_metal: top metal thickness.
    """
    return LayerStack(
        layers=dict(
            core=LayerLevel(
                layer=LogicalLayer(layer=LAYER.WG),
                thickness=thickness_wg,
                zmin=0.0,
                material="si",
                info={"mesh_order": 1},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
            slab=LayerLevel(
                layer=LogicalLayer(layer=LAYER.SLAB),
                thickness=thickness_slab,
                zmin=0.0,
                material="si",
                info={"mesh_order": 1},
                sidewall_angle=10,
                width_to_z=0.5,
            ),
            heater=LayerLevel(
                layer=LogicalLayer(layer=LAYER.HEATER),
                thickness=thickness_heater,
                zmin=zmin_heater,
                material="TiN",
                info={"mesh_order": 1},
            ),
            metal=LayerLevel(
                layer=LogicalLayer(layer=LAYER.PAD),
                thickness=thickness_metal,
                zmin=zmin_metal + thickness_metal,
                material="Aluminum",
                info={"mesh_order": 2},
            ),
        )
    )


LAYER_STACK = get_layer_stack()
LAYER_VIEWS = gf.technology.LayerViews(PATH.lyp_yaml)


class Tech:
    """Technology parameters."""

    radius = 5
    radius_strip = 5
    radius_rib = 25
    radius_ro = 25

    width = 0.45
    width_rib = 0.5
    width_ro = 0.5
    width_edge_coupler_tip = 0.2

    width_slab = 5
    width_heater = 2.5
    width_metal = 10

    gap_strip = 0.27
    gap_rib = 0.27


TECH = Tech()

############################
# Cross-sections functions
############################

cross_sections: dict[str, Callable[..., CrossSection]] = {}
_cross_section_default_names: dict[str, str] = {}


def xsection(func: Callable[..., CrossSection]) -> Callable[..., CrossSection]:
    """Returns decorated to register a cross section function.

    Ensures that the cross-section name matches the name of the function that generated it when created using default parameters

    .. code-block:: python

        @xsection
        def strip(width=TECH.width_strip, radius=TECH.radius_strip):
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


@xsection
def strip(
    width: float = TECH.width,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius,
    radius_min: float = TECH.radius,
) -> CrossSection:
    """Return Strip cross_section."""
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
    )


@xsection
def rib(
    width: float = TECH.width_rib,
    layer: LayerSpec = "WG",
    radius: float = TECH.radius_rib,
    radius_min: float = TECH.radius_rib,
    bbox_layers: LayerSpecs = ("SLAB",),
    bbox_offsets: Floats = (TECH.width_slab,),
    **kwargs,
) -> CrossSection:
    """Return Rib cross_section."""
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        bbox_layers=bbox_layers,
        bbox_offsets=bbox_offsets,
        **kwargs,
    )


@xsection
def strip_heater_metal(
    width: float = TECH.width,
    layer: LayerSpec = "WG",
    heater_width: float = TECH.width_heater,
    layer_heater: LayerSpec = "HEATER",
) -> CrossSection:
    """Returns strip cross_section with top heater metal."""
    return gf.cross_section.strip_heater_metal(
        width=width,
        layer=layer,
        heater_width=heater_width,
        layer_heater=layer_heater,
    )


@xsection
def metal_routing(
    width: float = 10,
    layer: LayerSpec = "PAD",
    radius: float | None = None,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names_electrical,
        port_types=port_types_electrical,
    )


@xsection
def heater_metal(width=TECH.width_heater) -> CrossSection:
    """Heater cross-section."""
    return gf.cross_section.heater_metal(width=width, layer=LAYER.HEATER)


############################
# Routing functions
############################

route_single = partial(gf.routing.route_single, cross_section="strip")
route_bundle = partial(gf.routing.route_bundle, cross_section="strip")


route_bundle_rib = partial(
    route_bundle,
    cross_section="rib",
)
route_bundle_metal = partial(
    route_bundle,
    straight="straight_metal",
    bend="bend_metal",
    taper=None,
    cross_section="metal_routing",
    port_type="electrical",
)
route_bundle_metal_corner = partial(
    route_bundle,
    straight="straight_metal",
    bend="wire_corner",
    taper=None,
    cross_section="metal_routing",
    port_type="electrical",
)

route_astar = partial(
    add_bundle_astar,
    layers=["WG"],
    bend="bend_euler",
    straight="straight",
    grid_unit=500,
    spacing=3,
)

route_astar_metal = partial(
    add_bundle_astar,
    layers=["PAD"],
    bend="wire_corner",
    straight="straight_metal",
    grid_unit=500,
    spacing=15,
)


routing_strategies = dict(
    route_bundle=route_bundle,
    route_bundle_rib=route_bundle_rib,
    route_bundle_metal=route_bundle_metal,
    route_bundle_metal_corner=route_bundle_metal_corner,
    route_astar=route_astar,
    route_astar_metal=route_astar_metal,
)
