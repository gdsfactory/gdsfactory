from __future__ import annotations

import pathlib
import tempfile
from functools import cache

import gdstk

from gdsfactory.component import Component, Port
from gdsfactory.read.import_gds import import_gds
from gdsfactory.typings import Layer


@cache
def from_gdstk(cell: gdstk.Cell, **kwargs) -> Component:
    """Returns gdsfactory Component from a gdstk cell.

    Args:
        cell: gdstk cell.

    Keyword Args:
        cellname: cell of the name to import (None) imports top cell.
        snap_to_grid_nm: snap to different nm grid (does not snap if False).
        gdsdir: optional GDS directory.
        read_metadata: loads metadata if it exists.
        hashed_name: appends a hash to a shortened component name.
        kwargs: extra to add to component.info (polarization, wavelength ...).
    """
    with tempfile.TemporaryDirectory() as gdsdir:
        gdsdir = pathlib.Path(gdsdir)
        gdsdir.mkdir(exist_ok=True)
        gdspath = gdsdir / f"{cell.name}.gds"
        filepath = cell.write_gds(gdspath)
        return import_gds(filepath, **kwargs)


@cache
def from_phidl(component, port_layer: Layer = (1, 0), **kwargs) -> Component:
    """Returns gdsfactory Component from a phidl Device or function.

    Args:
        component: phidl component.
        port_layer: to add to component ports.

    Keyword Args:
        cellname: cell of the name to import (None) imports top cell.
        snap_to_grid_nm: snap to different nm grid (does not snap if False).
        gdsdir: optional GDS directory.
        read_metadata: loads metadata if it exists.
        hashed_name: appends a hash to a shortened component name.
        kwargs: extra to add to component.info (polarization, wavelength ...).
    """
    device = component() if callable(component) else component

    with tempfile.TemporaryDirectory() as gdsdir:
        gdsdir = pathlib.Path(gdsdir)
        gdsdir.mkdir(exist_ok=True)
        gdspath = gdsdir / f"{device.name}.gds"
        filepath = device.write_gds(gdspath, cellname=device.name)

        component = import_gds(filepath, cache=False, **kwargs)
        component.unlock()

    for p in device.ports.values():
        if p.name not in component.ports:
            component.add_port(
                port=Port(
                    name=p.name,
                    center=p.midpoint,
                    width=p.width,
                    orientation=p.orientation,
                    parent=p.parent,
                    layer=port_layer,
                )
            )
    component.lock()
    return component


if __name__ == "__main__":
    import phidl.geometry as pg

    c = pg.rectangle()

    c = pg.snspd()
    c2 = from_phidl(component=c)
    c3 = from_phidl(component=c)
    c2.show(show_ports=True)
