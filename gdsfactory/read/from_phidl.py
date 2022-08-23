import pathlib
import tempfile

import gdspy
from phidl.device_layout import Device

from gdsfactory.component import Component, Port
from gdsfactory.read.import_gds import import_gds
from gdsfactory.types import Layer


def from_gdspy(cell: gdspy.Cell) -> Component:
    """Returns gdsfactory Component from a gdspy cell.

    Args:
        cell: gdspy cell.
    """
    with tempfile.TemporaryDirectory() as gdsdir:
        gdsdir = pathlib.Path(gdsdir)
        gdsdir.mkdir(exist_ok=True)
        gdspath = gdsdir / f"{cell.name}.gds"
        filepath = cell.write_gds(gdspath)
        component = import_gds(filepath)
        component.unlock()

    component.lock()
    return component


def from_phidl(component: Device, port_layer: Layer = (1, 0), **kwargs) -> Component:
    """Returns gdsfactory Component from a phidl Device or function.

    Args:
        component: phidl component.
        port_layer: to add to component ports.
        kwargs: ignore keyword args.
    """
    device = component() if callable(component) else component

    with tempfile.TemporaryDirectory() as gdsdir:
        gdsdir = pathlib.Path(gdsdir)
        gdsdir.mkdir(exist_ok=True)
        gdspath = gdsdir / f"{device.name}.gds"
        filepath = device.write_gds(gdspath, cellname=device.name)

        component = import_gds(filepath)
        component.unlock()

    for p in device.ports.values():
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
    print(c2.ports)
    c2.show(show_ports=True)
