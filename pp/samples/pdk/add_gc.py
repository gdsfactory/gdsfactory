import pp
from pp.routing.manhattan import round_corners
from pp.rotate import rotate

from import_gds import import_gds
from layers import LAYER
from waveguide import waveguide
from bend_circular import bend_circular


def gc_te1550():
    c = import_gds("ebeam_gc_te1550")
    c = rotate(c, 180)
    c.polarization = "te"
    c.wavelength = 1550
    return c


def gc_te1550_broadband():
    c = import_gds("ebeam_gc_te1550_broadband")
    return c


def gc_te1310():
    c = import_gds("ebeam_gc_te1310")
    c.polarization = "te"
    c.wavelength = 1310
    return c


def gc_tm1550():
    c = import_gds("ebeam_gc_tm1550")
    c.polarization = "tm"
    c.wavelength = 1550
    return c


def connect_strip(
    way_points=[],
    bend_factory=bend_circular,
    straight_factory=waveguide,
    bend_radius=10.0,
    wg_width=0.5,
    **kwargs,
):
    """
    Returns a deep-etched route formed by the given way_points with
    bends instead of corners and optionally tapers in straight sections.
    """
    bend90 = bend_factory(radius=bend_radius, width=wg_width)
    connector = round_corners(way_points, bend90, straight_factory)
    return connector


@pp.autoname
def taper_factory(layer=LAYER.WG, layers_cladding=[], **kwargs):
    c = pp.c.taper(layer=layer, layers_cladding=layers_cladding, **kwargs)
    return c


def add_gc(
    component,
    layer_label=LAYER.LABEL,
    grating_coupler=gc_te1550,
    bend_factory=bend_circular,
    straight_factory=waveguide,
    taper_factory=taper_factory,
    route_filter=connect_strip,
    gc_port_name="W0",
):
    c = pp.routing.add_io_optical(
        component,
        bend_factory=bend_factory,
        straight_factory=straight_factory,
        route_filter=route_filter,
        grating_coupler=grating_coupler,
        layer_label=layer_label,
        taper_factory=taper_factory,
        gc_port_name=gc_port_name,
    )
    c = rotate(c, -90)
    return c


if __name__ == "__main__":
    c = gc_te1550()
    print(c.ports)
    c = add_gc(component=waveguide())
    pp.show(c)
