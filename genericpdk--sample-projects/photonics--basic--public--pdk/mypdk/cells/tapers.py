"""Tapers."""

from functools import partial

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from mypdk.tech import LAYER, TECH


@gf.cell
def edge_coupler(
    length: float = 200,
    width1: float = TECH.width,
    width2: float = TECH.width_edge_coupler_tip,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        cross_section=cross_section,
        layer=None,
        port=None,
        with_two_ports=True,
        port_names=("o1", "o2"),
        port_types=("optical", "optical"),
        with_bbox=True,
    )


@gf.cell
def taper(
    length: float = 10.0,
    width1: float = TECH.width,
    width2: float | None = None,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        cross_section=cross_section,
        layer=None,
        port=None,
        with_two_ports=True,
        port_names=("o1", "o2"),
        port_types=("optical", "optical"),
        with_bbox=True,
    )


@gf.cell
def taper_metal(
    length: float = 10.0,
    width1: float = TECH.width_metal,
    width2: float | None = None,
    cross_section: CrossSectionSpec = "metal_routing",
) -> gf.Component:
    """Linear taper, which tapers only the main cross section section.

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
    """
    return gf.c.taper(
        length=length,
        width1=width1,
        width2=width2,
        cross_section=cross_section,
        layer=None,
        port=None,
        with_two_ports=True,
        port_names=("e1", "e2"),
        port_types=("electrical", "electrical"),
        with_bbox=True,
    )


@gf.cell
def taper_strip_to_ridge(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float = 0.5,
    w_slab1: float = 0.2,
    w_slab2: float = 10.45,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """A taper between strip and ridge.

    This is a transition between two distinct cross sections

    Args:
        length: the length of the taper.
        width1: the input width of the taper.
        width2: the output width of the taper.
        w_slab1: the input slab width of the taper.
        w_slab2: the output slab width of the taper.
        cross_section: a cross section or its name or a function generating a cross section.
    """
    return gf.c.taper_strip_to_ridge(
        length=length,
        width1=width1,
        width2=width2,
        w_slab1=w_slab1,
        w_slab2=w_slab2,
        cross_section=cross_section,
        layer_wg=LAYER.WG,
        layer_slab=LAYER.SLAB,
    )


trans_rib10 = partial(taper_strip_to_ridge, length=10)
trans_rib20 = partial(taper_strip_to_ridge, length=20)
trans_rib50 = partial(taper_strip_to_ridge, length=50)
