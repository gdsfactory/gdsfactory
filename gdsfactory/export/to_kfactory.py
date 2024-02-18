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


def to_kfactory(component) -> kf.KCell:
    """Converts a gdsfactory component to a kfactory component."""
    c0 = component.write_gds(with_metadata=True)
    c = import_gds(c0)

    for k, v in dict(component.info).items():
        c.info[k] = v
    for k, v in dict(component.settings).items():
        if isinstance(v, str | int | float | bool):
            c.info[k] = v
    for port in component.ports.values():
        center = port.center
        orientation = port.orientation
        layer = gf.get_layer(port.layer)
        layer = kf.kcl.layer(*layer)
        trans = kf.kdb.DCplxTrans(1, orientation, False, center[0], center[1])
        width = port.width
        c.create_port(
            name=port.name,
            dwidth=round(width * 1e3) / 1e3,
            layer=layer,
            port_type=port.port_type,
            dcplx_trans=trans,
        )
    return c


if __name__ == "__main__":
    c0 = gf.components.mzi()
    c0 = gf.pack([gf.c.mzi(), gf.c.straight()])[0]
    c = to_kfactory(c0)
    c.show()
    # c = gf.components.mzi()
    # c.write_gds(with_metadata=True)
