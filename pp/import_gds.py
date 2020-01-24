import os
import gdspy

from phidl.device_layout import DeviceReference

import pp
from pp.component import NAME_TO_DEVICE


def import_gds(filename, cellname=None, flatten=False, overwrite_cache=False):
    """ returns a Componenent from a GDS file
    """
    filename = str(filename)
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(filename)
    top_level_cells = gdsii_lib.top_level()

    if cellname is not None:
        if cellname not in gdsii_lib.cell_dict:
            raise ValueError(
                "[PHIDL] import_gds() The requested cell (named %s) is not present in file %s"
                % (cellname, filename)
            )
        topcell = gdsii_lib.cell_dict[cellname]
    elif cellname is None and len(top_level_cells) == 1:
        topcell = top_level_cells[0]
    elif cellname is None and len(top_level_cells) > 1:
        raise ValueError(
            "[PHIDL] import_gds() There are multiple top-level cells in {}, you must specify `cellname` to select of one of them among {}".format(
                filename, [_c.name for _c in top_level_cells]
            )
        )

    if flatten == False:
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
                except:
                    print("WARNING - Could not import", e.ref_cell.name)

            D.references = converted_references
            # Next convert each Polygon
            temp_polygons = list(D.polygons)
            D.polygons = []
            for p in temp_polygons:
                D.add_polygon(p)
                # else:
                #     warnings.warn('[PHIDL] import_gds(). Warning an element which was not a ' \
                #         'polygon or reference exists in the GDS, and was not able to be imported. ' \
                #         'The element was a: "%s"' % e)

        topdevice = c2dmap[topcell.name]
        return topdevice

    elif flatten == True:
        D = pp.Component()
        polygons = topcell.get_polygons(by_spec=True)

        for layer_in_gds, polys in polygons.items():
            D.add_polygon(polys, layer=layer_in_gds)
        return D


if __name__ == "__main__":

    filename = os.path.join(pp.CONFIG["gdslib"], "mzi2x2.gds")
    filename = pp.CONFIG["gdslib"] / "mzi2x2.gds"
    c = import_gds(filename)
    print(c)
    pp.show(c)
