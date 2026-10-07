"""Waveguide crossings."""

from __future__ import annotations

__all__ = [
    "crossing",
    "crossing45",
    "crossing_arm",
    "crossing_etched",
    "crossing_linear_taper",
]

from kfactory.conf import CheckInstances

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Delta, LayerSpec

from .._schematic import crossing_schematic


@gf.cell_with_module_name(tags=["waveguides"])
def crossing_arm(
    r1: float = 3.0,
    r2: float = 1.1,
    w: float = 1.2,
    L: float = 3.4,
    layer_slab: LayerSpec = "SLAB150",
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns crossing arm.

    Args:
        r1: ellipse radius1.
        r2: ellipse radius2.
        w: width in um.
        L: length in um.
        layer_slab: for the shallow etch.
        cross_section: spec.
    """
    return cf.crossing_arm(
        r1=r1, r2=r2, w=w, L=L, layer_slab=layer_slab, cross_section=cross_section
    )


@gf.cell_with_module_name(schematic_function=crossing_schematic, tags=["waveguides"])
def crossing(
    arm: ComponentSpec = "crossing_arm",
) -> gf.Component:
    """Waveguide crossing.

    Args:
        arm: arm spec.
    """
    return cf.crossing(arm=arm)


@gf.cell_with_module_name(schematic_function=crossing_schematic, tags=["waveguides"])
def crossing_linear_taper(
    width1: float = 2.5,
    width2: float = 0.5,
    length: float = 3,
    cross_section: CrossSectionSpec = "strip",
    taper: ComponentSpec = "taper",
) -> Component:
    """Returns Crossing based on a taper.

    The default is a dummy taper.

    Args:
        width1: input width.
        width2: output width.
        length: taper length.
        cross_section: cross_section spec.
        taper: taper spec.
    """
    return cf.crossing_linear_taper(
        width1=width1,
        width2=width2,
        length=length,
        cross_section=cross_section,
        taper=taper,
    )


@gf.cell_with_module_name(schematic_function=crossing_schematic, tags=["waveguides"])
def crossing_etched(
    width: float = 0.5,
    r1: float = 3.0,
    r2: float = 1.1,
    w: float = 1.2,
    L: float = 3.4,
    layer_wg: LayerSpec = "WG",
    layer_slab: LayerSpec = "SLAB150",
) -> Component:
    """Waveguide crossing.

    Full crossing has to be on WG layer (to start with a 220nm slab).
    Then we etch the ellipses down to 150nm slabs and we keep linear taper at 220nm.

    Args:
        width: input waveguides width.
        r1: radii.
        r2: radii.
        w: wide width.
        L: length.
        layer_wg: waveguide layer.
        layer_slab: shallow etch layer.
    """
    return cf.crossing_etched(
        width=width, r1=r1, r2=r2, w=w, L=L, layer_wg=layer_wg, layer_slab=layer_slab
    )


@gf.cell(
    check_instances=CheckInstances.IGNORE,
    with_module_name=True,
    schematic_function=crossing_schematic,
    tags=["waveguides"],
)
def crossing45(
    crossing: ComponentSpec = "crossing",
    port_spacing: float = 40.0,
    dx: Delta | None = None,
    alpha: float = 0.08,
    npoints: int = 101,
    cross_section: CrossSectionSpec = "strip",
    cross_section_bends: CrossSectionSpec = "strip",
) -> Component:
    r"""Returns 45deg crossing with bends.

    Args:
        crossing: crossing function.
        port_spacing: target I/O port spacing.
        dx: target length.
        alpha: optimization parameter. diminish it for tight bends,
          increase it if raises assertion angle errors
        npoints: number of points.
        cross_section: cross_section spec.
        cross_section_bends: cross_section spec.


    The 45 Degree crossing CANNOT be kept as an SRef since
    we only allow for multiples of 90Deg rotations in SRef.

    ```text
        ----   ----
            \ /
             X
            / \
        ---    ----
    ```

    """
    return cf.crossing45(
        crossing=crossing,
        port_spacing=port_spacing,
        dx=dx,
        alpha=alpha,
        npoints=npoints,
        cross_section=cross_section,
        cross_section_bends=cross_section_bends,
    )


__all__ = [
    "crossing",
    "crossing45",
    "crossing_arm",
    "crossing_etched",
    "crossing_linear_taper",
]
