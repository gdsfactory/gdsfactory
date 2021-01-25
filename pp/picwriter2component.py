"""Picwriter is a photonics library written by Derek Kita.

https://picwriter.readthedocs.io/en/latest/component-documentation.html
As it is based on gdspy it's easier to wrap picwriter components
"""
import gdspy
import numpy as np
import picwriter.components as pc
import picwriter.toolkit as pt
from picwriter.toolkit import Component

import pp

gdspy.current_library = gdspy.GdsLibrary()

direction_to_orientation = dict(EAST=0.0, NORTH=90.0, WEST=180.0, SOUTH=270.0)


def direction_to_degree(direction: str) -> float:
    """Convert a 'direction' (as used in picwriter) to an angle in degrees.
    picwriter 'direction's can be either a float (corresponding to an angle in radians)
    or a string, corresponding to a cardinal direction
    """
    if isinstance(direction, float):
        # direction is a float in radians, but rotation should be a float in degrees
        return direction * 180.0 / np.pi
    return direction_to_orientation[direction]


def picwriter2component(picwriter_object: pt.Component) -> Component:
    """Convert a Picwriter into a Gdsfactory Component."""
    po = picwriter_object
    c = pp.Component(name=po.name_prefix)

    # Extract the relevant cells from the picwriter global cell list
    po_cell = pt.CURRENT_CELLS[po.cell_hash]

    polygons = po_cell.get_polygonsets()
    for poly in polygons:
        polygons = poly.polygons
        layers = poly.layers
        datatypes = poly.datatypes

        for polygon, layer, datatype in zip(polygons, layers, datatypes):
            c.add_polygon(polygon, layer=(layer, datatype))

    translate_by = po.port
    rotate_by = direction_to_degree(po.direction)

    c.rotate(rotate_by)  # First rotate about (0,0)
    c.move(translate_by)  # Next translate

    for port in po.portlist.keys():
        port_loc = po.portlist[port]["port"]
        direction = direction_to_degree(po.portlist[port]["direction"])

        c.add_port(
            name=port,
            midpoint=[port_loc[0], port_loc[1]],
            width=po.wgt.wg_width,
            orientation=direction,
        )

    return c


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
    gcc = picwriter2component(gc)

    pp.show(gcc)
