from pathlib import Path

import kfactory as kf

import gdsfactory as gf


def import_gds(gdspath: str | Path) -> kf.KCell:
    """Reads a GDS file and returns a kfactory component.

    Args:
        gdspath: path to GDS file.
    """
    read_options = kf.kcell.load_layout_options()
    read_options.cell_conflict_resolution = (
        kf.kdb.LoadLayoutOptions.CellConflictResolution.RenameCell
    )
    top_cells = set(kf.kcl.top_cells())
    kf.kcl.read(gdspath, read_options, test_merge=True)
    new_top_cells = set(kf.kcl.top_cells()) - top_cells
    if len(new_top_cells) != 1:
        raise ValueError(f"Expected 1 new top cell, got {len(new_top_cells)}")
    return kf.kcl[new_top_cells.pop().name]


def add_ports(c: kf.KCell, gf_component: gf.Component):
    """Adds ports to a kfactory component.

    Args:
        c: kfactory component.
        gf_component: gdsfactory component.
    """
    for port in gf_component.ports.values():
        center = port.center
        orientation = port.orientation or 0
        layer = gf.get_layer(port.layer)
        layer = kf.kcl.layer(*layer)
        trans = kf.kdb.DCplxTrans(1, float(orientation), False, center[0], center[1])
        width = port.width
        c.create_port(
            name=port.name,
            dwidth=round(width * 1e3) / 1e3,
            layer=layer,
            port_type=port.port_type,
            dcplx_trans=trans,
        )


def to_kfactory(component: gf.Component, recursive: bool = True) -> kf.KCell:
    """Converts a gdsfactory component to a kfactory component.

    Args:
        component: gdsfactory component.
        recursive: if True, recursively add ports and info to all subcomponents.

    """
    c0 = component.write_gds(with_metadata=True)
    c = import_gds(c0)

    for k, v in sorted(dict(component.info).items()):
        c.info[k] = v

    settings = {}
    for k, v in sorted(dict(component.settings).items()):
        if isinstance(v, str | int | float | bool):
            settings[k] = v

    c._settings = kf.kcell.KCellSettings(**settings)
    add_ports(c, component)

    if recursive:
        for child in component.references:
            component = child.parent
            cell_name = component.name
            kcell_child = kf.kcl[cell_name]
            add_ports(kcell_child, component)

            for k, v in sorted(dict(component.info).items()):
                kcell_child.info[k] = v

            settings = {}
            for k, v in sorted(dict(component.settings).items()):
                if isinstance(v, str | int | float | bool):
                    settings[k] = v
            kcell_child._settings = kf.kcell.KCellSettings(**settings)

    return c


if __name__ == "__main__":
    # c0 = gf.components.mzi()
    c0 = gf.pack([gf.c.mzi(), gf.c.straight()])[0]
    c = to_kfactory(c0)
    c.show()
    # c = gf.components.mzi()
    # c.write_gds(with_metadata=True)
