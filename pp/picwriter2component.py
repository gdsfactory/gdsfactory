""" Picwriter is a photonics library written by Derek Kita
https://picwriter.readthedocs.io/en/latest/component-documentation.html
"""
import numpy as np
import gdspy

import picwriter.toolkit as pt
import picwriter.components as pc
import pp
from pp.component import Component


gdspy.current_library = gdspy.GdsLibrary()


def direction_to_degree(direction: str) -> float:
    """ Converts a 'direction' (as used in picwriter) to an angle in degrees.
    picwriter 'direction's can be either a float (corresponding to an angle in radians)
    or a string, corresponding to a cardinal direction
    """
    if isinstance(direction, float):
        # direction is a float in radians, but rotation should be a float in degrees
        return direction * 180.0 / np.pi
    elif str(direction) == "EAST":
        return 0.0
    elif str(direction) == "NORTH":
        return 90.0
    elif str(direction) == "WEST":
        return 180.0
    elif str(direction) == "SOUTH":
        return 270.0


def picwriter2component(picwriter_object: pt.Component) -> Component:
    """ Converts a Picwriter into a Gdsfactory Component
    """
    po = picwriter_object
    c = pp.Component(name=po.name_prefix)

    # Add the polygons
    po_cell = pt.CURRENT_CELLS[
        po.cell_hash
    ]  # Extract the relevant cells from the picwriter global cell list

    ps = po_cell.get_polygonsets()
    for i in range(len(ps)):
        polygons = ps[i].polygons
        layers = ps[i].layers
        datatypes = ps[i].datatypes

        for j in range(len(polygons)):
            c.add_polygon(polygons[j], layer=(layers[j], datatypes[j]))

    translate_by = po.port
    rotate_by = direction_to_degree(po.direction)

    c.rotate(rotate_by)  # First rotate about (0,0)
    c.move(translate_by)  # Next translate

    """ Add the ports """
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
