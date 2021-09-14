"""Picwriter is a photonics library written by Derek Kita.

https://picwriter.readthedocs.io/en/latest/component-documentation.html
As it is based on gdspy it's easier to wrap picwriter components
"""
import gdspy
import numpy as np
import picwriter.components as pc
import picwriter.toolkit as pt
from picwriter.components.waveguide import WaveguideTemplate
from picwriter.toolkit import Component

import gdsfactory as gf
from gdsfactory.types import CrossSectionFactory, Layer

gdspy.current_library = gdspy.GdsLibrary()

direction_to_orientation = dict(EAST=0.0, NORTH=90.0, WEST=180.0, SOUTH=270.0)


def cross_section_to_waveguide_template(
    cross_section: CrossSectionFactory,
    euler_bend: bool = True,
    wg_type: str = "strip",
    **kwargs
) -> WaveguideTemplate:
    x = cross_section(**kwargs)

    layer = x.info["layer"]
    layer_cladding = x.info["layers_cladding"][0]

    return pc.WaveguideTemplate(
        bend_radius=x.info["radius"],
        wg_width=x.info["width"],
        wg_layer=layer[0],
        wg_datatype=layer[1],
        clad_layer=layer_cladding[0],
        clad_datatype=layer_cladding[1],
        clad_width=x.info["cladding_offset"],
        wg_type=wg_type,
        euler_bend=euler_bend,
    )


def direction_to_degree(direction: str) -> float:
    """Convert a 'direction' (as used in picwriter) to an angle in degrees.
    picwriter 'direction's can be either a float (corresponding to an angle in radians)
    or a string, corresponding to a cardinal direction
    """
    if isinstance(direction, float):
        # direction is a float in radians, but rotation should be a float in degrees
        return direction * 180.0 / np.pi
    return direction_to_orientation[direction]


def picwriter(picwriter_object: pt.Component, port_layer: Layer = (1, 0)) -> Component:
    """Convert a Picwriter into a Gdsfactory Component."""
    po = picwriter_object
    c = gf.Component(name=po.name_prefix)

    # Extract the relevant cells from the picwriter global cell list
    po_cell = pt.CURRENT_CELLS[po.cell_hash]

    polygons = po_cell.get_polygonsets()
    for poly in polygons:
        polygons = poly.polygons
        layers = poly.layers
        datatypes = poly.datatypes

        for polygon, layer, datatype in zip(polygons, layers, datatypes):
            c.add_polygon(polygon, layer=(layer, datatype))

    c2 = gf.Component()
    ref = c2.add_ref(c)

    translate_by = po.port
    rotate_by = direction_to_degree(po.direction)
    ref.rotate(rotate_by)  # First rotate about (0,0)
    ref.move(translate_by)  # Next translate

    for port in po.portlist.keys():
        port_loc = po.portlist[port]["port"]
        direction = direction_to_degree(po.portlist[port]["direction"])

        c2.add_port(
            name=port,
            midpoint=[port_loc[0], port_loc[1]],
            width=po.wgt.wg_width,
            orientation=direction,
            layer=port_layer,
        )

    c2.absorb(ref)
    c2.auto_rename_ports()
    return c2


if __name__ == "__main__":

    wgt = pc.WaveguideTemplate(
        bend_radius=50.0,
        wg_width=1.0,
        wg_layer=1,
        wg_datatype=0,
        clad_layer=2,
        clad_datatype=0,
    )

    # gc = pc.GratingCoupler(wgt, port=(10, 20), direction=np.pi * 7 / 8)
    gc = pc.GratingCoupler(wgt, port=(10, 20), direction=0.0)
    gcc = picwriter(gc)

    gf.show(gcc)
