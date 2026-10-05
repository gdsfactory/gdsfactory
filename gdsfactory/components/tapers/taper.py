from __future__ import annotations

__all__ = [
    "taper",
    "taper_electrical",
    "taper_nc_sc",
    "taper_sc_nc",
    "taper_strip_to_ridge",
    "taper_strip_to_ridge_trenches",
    "taper_strip_to_slab150",
]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.component_functions import CellAlias
from gdsfactory.port import Port
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from .._schematic import taper_schematic, transition_schematic


@gf.cell_with_module_name(schematic_function=taper_schematic, tags=["tapers"])
def taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    layer: LayerSpec | None = None,
    port: Port | None = None,
    with_two_ports: bool = True,
    cross_section: CrossSectionSpec = "strip",
    port_names: tuple[str, str] = ("o1", "o2"),
    port_types: tuple[str, str] = ("optical", "optical"),
    with_bbox: bool = True,
) -> Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        layer: layer for the taper.
        port: can taper from a port instead of defining width1.
        with_two_ports: includes a second port.
            False for terminator and edge coupler fiber interface.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        port_names: input and output port names. Second name only used if with_two_ports.
        port_types: input and output port types. Second type only used if with_two_ports.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
    """
    return cf.taper(
        length=length,
        width1=width1,
        width2=width2,
        layer=layer,
        port=port,
        with_two_ports=with_two_ports,
        cross_section=cross_section,
        port_names=port_names,
        port_types=port_types,
        with_bbox=with_bbox,
    )


@gf.cell_with_module_name(schematic_function=transition_schematic, tags=["tapers"])
def taper_strip_to_ridge(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float = 0.5,
    w_slab1: float = 0.15,
    w_slab2: float = 6.0,
    layer_wg: LayerSpec = "WG",
    layer_slab: LayerSpec = "SLAB90",
    cross_section: CrossSectionSpec = "strip",
    use_slab_port: bool = False,
    slab_port_layer: LayerSpec | None = None,
) -> Component:
    r"""Linear taper from strip to rib.

    Args:
        length: taper length (um).
        width1: in um.
        width2: in um.
        w_slab1: slab width in um.
        w_slab2: slab width in um.
        layer_wg: for input waveguide.
        layer_slab: for output waveguide with slab.
        cross_section: for input waveguide.
        use_slab_port: if True adds a second port for the slab.
        slab_port_layer: if specified, overrides the layer for the slab port.

    ```text
                      __________________________
                     /           |
             _______/____________|______________
                   /             |
       width1     |w_slab1       | w_slab2  width2
             ______\_____________|______________
                    \            |
                     \__________________________
    ```

    """
    return cf.taper_strip_to_ridge(
        length=length,
        width1=width1,
        width2=width2,
        w_slab1=w_slab1,
        w_slab2=w_slab2,
        layer_wg=layer_wg,
        layer_slab=layer_slab,
        cross_section=cross_section,
        use_slab_port=use_slab_port,
        slab_port_layer=slab_port_layer,
    )


@gf.cell_with_module_name(schematic_function=transition_schematic, tags=["tapers"])
def taper_strip_to_ridge_trenches(
    length: float = 10.0,
    width: float = 0.5,
    slab_offset: float = 3.0,
    trench_width: float = 2.0,
    trench_layer: LayerSpec = "DEEP_ETCH",
    layer_wg: LayerSpec = "WG",
    trench_offset: float = 0.1,
) -> gf.Component:
    """Defines taper using trenches to define the etch.

    Args:
        length: in um.
        width: in um.
        slab_offset: in um.
        trench_width: in um.
        trench_layer: trench layer.
        layer_wg: waveguide layer.
        trench_offset: after waveguide in um.
    """
    return cf.taper_strip_to_ridge_trenches(
        length=length,
        width=width,
        slab_offset=slab_offset,
        trench_width=trench_width,
        trench_layer=trench_layer,
        layer_wg=layer_wg,
        trench_offset=trench_offset,
    )


taper_strip_to_slab150 = CellAlias(taper_strip_to_ridge, layer_slab="SLAB150")


@gf.cell_with_module_name(schematic_function=transition_schematic, tags=["tapers"])
def taper_sc_nc(
    width1: float = 0.5,
    width2: float = 1,
    length: float = 20,
    layer_wg: LayerSpec = "WG",
    layer_nitride: LayerSpec = "WGN",
    width_tip_nitride: float = 0.15,
    width_tip_silicon: float = 0.15,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Taper from strip to nitride.

    Args:
        width1: strip width.
        width2: nitride width.
        length: taper length.
        layer_wg: strip layer.
        layer_nitride: nitride layer.
        width_tip_nitride: tip width for nitride.
        width_tip_silicon: tip width for strip.
        cross_section: cross_section specification.
    """
    return cf.taper_sc_nc(
        width1=width1,
        width2=width2,
        length=length,
        layer_wg=layer_wg,
        layer_nitride=layer_nitride,
        width_tip_nitride=width_tip_nitride,
        width_tip_silicon=width_tip_silicon,
        cross_section=cross_section,
    )


@gf.cell_with_module_name(schematic_function=transition_schematic, tags=["tapers"])
def taper_nc_sc(
    width1: float = 1,
    width2: float = 0.5,
    length: float = 20,
    layer_wg: LayerSpec = "WG",
    layer_nitride: LayerSpec = "WGN",
    width_tip_nitride: float = 0.15,
    width_tip_silicon: float = 0.15,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Taper from nitride to strip.

    Args:
        width1: nitride width.
        width2: silicon width.
        length: taper length.
        layer_wg: nitride layer.
        layer_nitride: strip layer.
        width_tip_nitride: tip width for nitride.
        width_tip_silicon: tip width for strip.
        cross_section: cross_section specification.
    """
    return cf.taper_nc_sc(
        width1=width1,
        width2=width2,
        length=length,
        layer_wg=layer_wg,
        layer_nitride=layer_nitride,
        width_tip_nitride=width_tip_nitride,
        width_tip_silicon=width_tip_silicon,
        cross_section=cross_section,
    )


taper_electrical = CellAlias(
    taper,
    port_types=("electrical", "electrical"),
    port_names=("e1", "e2"),
    cross_section="metal_routing",
)

taper_with_trenches = CellAlias(
    taper,
    cross_section="rib_with_trenches",
)
