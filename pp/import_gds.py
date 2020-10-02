from typing import Optional
import gdspy

from phidl.device_layout import DeviceReference

import pp
from pp.component import Component
from pp.name import NAME_TO_DEVICE
from pp.port import read_port_markers, auto_rename_ports
from pp.layers import port_layer2type, port_type2layer


def add_ports_from_markers_inside(*args, **kwargs):
    """markers inside the device"""
    return add_ports_from_markers_center(inside=True, *args, **kwargs)


def add_ports_from_markers_center(
    component: Component,
    port_layer2type=port_layer2type,
    port_type2layer=port_type2layer,
    inside: bool = False,
    tol=0.1,
):
    """add ports from polygons in certain layers

    markers at port center, so half of the marker goes inside and half ouside the port

    Args:
        component: to read polygons from and to write ports to
        port_layer2type: dict of layer to port_type
        port_type2layer: dict of port_type to layer
        inside: True-> markers  inside. False-> markers at center

    .. code::
           _______________
          |               |
          |               |
          |               |
         |||              |
         |||              |
          |               |
          |      __       |
          |_______________|
                 __

    dx < dy: port is east or west
        x > xc: east
        x < xc: west

    dx > dy: port is north or south
        y > yc: north
        y < yc: south

    dx = dy
        x > xc: east
        x < xc: west

    """
    i = 0
    xc = component.x
    yc = component.y
    xmax = component.xmax
    xmin = component.xmin
    ymax = component.ymax
    ymin = component.ymin

    for port_layer, port_type in port_layer2type.items():
        port_markers = read_port_markers(component, [port_layer])

        for p in port_markers.polygons:
            dy = p.ymax - p.ymin
            dx = p.xmax - p.xmin
            x = p.x
            y = p.y
            layer = port_type2layer[port_type]
            pxmax = p.xmax
            pxmin = p.xmin
            pymax = p.ymax
            pymin = p.ymin

            if dx < dy and x > xc:  # east
                orientation = 0
                width = dy
                if inside:
                    x = p.xmax
            elif dx < dy and x < xc:  # west
                orientation = 180
                width = dy
                if inside:
                    x = p.xmin
            elif dx > dy and y > yc:  # north
                orientation = 90
                width = dx
                if inside:
                    y = p.ymax
            elif dx > dy and y < yc:  # south
                orientation = 270
                width = dx
                if inside:
                    y = p.ymin
            # port markers have same width and height
            # check which edge (E, W, N, S) they are closer to
            elif pxmax > xmax - tol:  # east
                orientation = 0
                width = dy
                x = p.xmax
            elif pxmin < xmin + tol:  # west
                orientation = 180
                width = dy
                x = p.xmin
            elif pymax > ymax - tol:  # north
                orientation = 90
                width = dx
                y = p.ymax
            elif pymin < ymin + tol:  # south
                orientation = 270
                width = dx
                y = p.ymin
            else:
                raise ValueError(f"port marker x={x} y={y}, dx={dx}, dy={dy}")

            component.add_port(
                i,
                midpoint=(x, y),
                width=width,
                orientation=orientation,
                port_type=port_type,
                layer=layer,
            )
            i += 1


def import_gds_cells(gdspath):
    """ returns top cells from GDS"""
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
    top_level_cells = gdsii_lib.top_level()
    return top_level_cells


def import_gds(
    gdspath: str,
    cellname: None = None,
    flatten: bool = False,
    overwrite_cache: bool = True,
    snap_to_grid_nm: Optional[int] = None,
) -> Component:
    """returns a Componenent from a GDS file

    Args:
        gdspath: path of GDS file
        cellname: cell of the name to import (None) imports top cell
        flatten: if True returns flattened (no hierarchy)
        overwrite_cache: overwrites device cache (caching by name)
        snap_to_grid_nm: snap

    """
    gdspath = str(gdspath)
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(gdspath)
    top_level_cells = gdsii_lib.top_level()
    cellnames = [c.name for c in top_level_cells]

    if cellname is not None:
        if cellname not in cellnames:
            raise ValueError(
                f"import_gds() The requested cell {cellname} is not present in file {gdspath} with cells {cellnames}"
            )
        topcell = gdsii_lib.cells[cellname]
    elif cellname is None and len(top_level_cells) == 1:
        topcell = top_level_cells[0]
    elif cellname is None and len(top_level_cells) > 1:
        raise ValueError(
            "import_gds() There are multiple top-level cells in {}, you must specify `cellname` to select of one of them among {}".format(
                gdspath, [_c.name for _c in top_level_cells]
            )
        )

    if flatten:
        D = pp.Component()
        polygons = topcell.get_polygons(by_spec=True)

        for layer_in_gds, polys in polygons.items():
            D.add_polygon(polys, layer=layer_in_gds)
        return D

    else:
        D_list = []
        c2dmap = {}
        all_cells = topcell.get_dependencies(True)
        all_cells.update([topcell])

        for cell in all_cells:
            cell_name = cell.name
            if overwrite_cache or cell_name not in NAME_TO_DEVICE:
                D = pp.Component()
                D.name = cell.name
                D.polygons = cell.polygons
                D.references = cell.references
                D.name = cell_name
                D.labels = cell.labels
            else:
                D = NAME_TO_DEVICE[cell_name]

            c2dmap.update({cell_name: D})
            D_list += [D]

        for D in D_list:
            # First convert each reference so it points to the right Device
            converted_references = []
            for e in D.references:
                try:
                    ref_device = c2dmap[e.ref_cell.name]

                    dr = DeviceReference(
                        device=ref_device,
                        origin=e.origin,
                        rotation=e.rotation,
                        magnification=e.magnification,
                        x_reflection=e.x_reflection,
                    )
                    converted_references.append(dr)
                except Exception:
                    print("WARNING - Could not import", e.ref_cell.name)

            D.references = converted_references
            # Next convert each Polygon
            temp_polygons = list(D.polygons)
            D.polygons = []
            for p in temp_polygons:
                if snap_to_grid_nm:
                    points_on_grid = pp.drc.snap_to_grid(
                        p.polygons[0], nm=snap_to_grid_nm
                    )
                    p = gdspy.Polygon(
                        points_on_grid, layer=p.layers[0], datatype=p.datatypes[0]
                    )
                D.add_polygon(p)
                # else:
                #     warnings.warn('[PHIDL] import_gds(). Warning an element which was not a ' \
                #         'polygon or reference exists in the GDS, and was not able to be imported. ' \
                #         'The element was a: "%s"' % e)

        topdevice = c2dmap[topcell.name]
        return topdevice


def test_import_gds_snap_to_grid():
    gdspath = pp.CONFIG["gdsdir"] / "mmi1x2.gds"
    c = import_gds(gdspath, snap_to_grid_nm=5)
    print(len(c.get_polygons()))
    assert len(c.get_polygons()) == 8

    for x, y in c.get_polygons()[0]:
        assert pp.drc.on_grid(x, 5)
        assert pp.drc.on_grid(y, 5)


def test_import_gds_hierarchy():
    c0 = pp.c.mzi2x2()
    gdspath = pp.write_gds(c0)
    c = import_gds(gdspath)
    assert len(c.get_dependencies()) == 3


def test_import_gds_with_port_markers_optical():
    """ """
    # c  =  pp.c.mmi1x2()
    # for p in c.ports.values():
    #     print(p)
    # pp.show(c)

    gdspath = pp.CONFIG["gdsdir"] / "mmi1x2.gds"
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    auto_rename_ports(c)

    # for p in c.ports.values():
    #     print(p)


def test_import_gds_with_port_markers_optical_electrical():
    """"""
    # c  =  pp.c.mzi2x2(with_elec_connections=True)
    # for p in c.ports.values():
    #     print(p)
    # pp.show(c)

    gdspath = pp.CONFIG["gdsdir"] / "mzi2x2.gds"
    c = import_gds(gdspath)
    add_ports_from_markers_center(c)
    auto_rename_ports(c)
    pp.show(c)

    for p in c.ports.values():
        print(p)


if __name__ == "__main__":
    test_import_gds_with_port_markers_optical_electrical()
    # test_import_gds_with_port_markers_optical()
    # test_import_gds_snap_to_grid()
    # test_import_gds_snap_to_grid()

    # gdspath = pp.CONFIG["gdslib"] / "gds" / "mzi2x2.gds"
    # c = import_gds(gdspath, snap_to_grid_nm=5)
    # print(c)
    # pp.show(c)
