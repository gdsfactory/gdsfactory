from __future__ import annotations

import pathlib
import tempfile
import uuid

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentOrPath

valid_operations = ("xor", "not", "and", "or")


@gf.cell
def boolean_klayout(
    gdspath1: ComponentOrPath,
    gdspath2: ComponentOrPath,
    layer1: tuple[int, int] = (1, 0),
    layer2: tuple[int, int] = (1, 0),
    layer3: tuple[int, int] = (2, 0),
    operation: str = "xor",
) -> Component:
    """Returns a boolean operation between two components Uses KLayout python API.

    Args:
        gdspath1: path to GDS or Component.
        gdspath2: path to GDS or Component.
        layer1: tuple for gdspath1.
        layer2: tuple for gdspath2.
        layer3: for the result of the operation.

    """
    import klayout.db as pya

    if operation not in valid_operations:
        raise ValueError(f"{operation} not in {valid_operations}")

    if isinstance(gdspath1, Component):
        gdspath1.flatten()
        gdspath1 = gdspath1.write_gds()

    if isinstance(gdspath2, Component):
        gdspath2.flatten()
        gdspath2 = gdspath2.write_gds()

    layout1 = pya.Layout()
    layout1.read(str(gdspath1))
    cell1 = layout1.top_cell()

    layout2 = pya.Layout()
    layout2.read(str(gdspath2))
    cell2 = layout2.top_cell()

    cellname = f"boolean_{str(uuid.uuid4())[:8]}"
    layout3 = pya.Layout()
    layout3_top = layout3.create_cell(cellname)

    a = pya.Region(cell1.begin_shapes_rec(layout1.layer(layer1[0], layer1[1])))
    b = pya.Region(cell2.begin_shapes_rec(layout2.layer(layer2[0], layer2[1])))

    if operation == "xor":
        result = a ^ b
    elif operation == "not":
        result = a - b
    elif operation == "and":
        result = a & b
    elif operation == "or":
        result = a | b

    layout3_top.shapes(layout3.layer(layer3[0], layer3[1])).insert(result)

    dirpath_build = pathlib.Path(tempfile.TemporaryDirectory().name)
    dirpath_build.mkdir(exist_ok=True, parents=True)
    gdspath = str(dirpath_build / f"{cellname}.gds")
    layout3.write(gdspath)
    return gf.import_gds(gdspath)


def _demo() -> None:
    import klayout.db as pya

    import gdsfactory as gf

    gdspath1 = gf.Component("ellipse1")
    gdspath1.add_ref(gf.components.ellipse(radii=[10, 5], layer=(1, 0)))

    gdspath2 = gf.Component("ellipse2")
    gdspath2.add_ref(gf.components.ellipse(radii=[11, 4], layer=(1, 0))).movex(4)

    layer1 = layer2 = (1, 0)
    layer3 = (1, 0)

    if isinstance(gdspath1, Component):
        gdspath1.flatten()
        gdspath1 = gdspath1.write_gds()

    if isinstance(gdspath2, Component):
        gdspath2.flatten()
        gdspath2 = gdspath2.write_gds()
        gf.show(gdspath2)

    layout1 = pya.Layout()
    layout1.read(str(gdspath1))
    cell1 = layout1.top_cell()

    layout2 = pya.Layout()
    layout2.read(str(gdspath2))
    cell2 = layout2.top_cell()

    layout3 = pya.Layout()
    layout3_top = layout3.create_cell("top")

    a = pya.Region(cell1.begin_shapes_rec(layout1.layer(layer1[0], layer1[1])))
    b = pya.Region(cell2.begin_shapes_rec(layout2.layer(layer2[0], layer2[1])))
    rxor = a ^ b
    layout3_top.shapes(layout3.layer(layer3[0], layer3[1])).insert(rxor)

    layout3_top.write("boolean.gds")
    gf.show("boolean.gds")


def _show_shapes() -> None:
    c1 = gf.components.ellipse(radii=[8, 8], layer=(1, 0))
    c2 = gf.components.ellipse(radii=[11, 4], layer=(1, 0))
    c3 = gf.Component()
    c3 << c1
    c3 << c2
    c3.show()


if __name__ == "__main__":
    # _show_shapes()
    c1 = gf.components.ellipse(radii=[8, 8], layer=(1, 0))
    c2 = gf.components.ellipse(radii=[11, 4], layer=(1, 0))
    c = boolean_klayout(c1, c2, operation="not")
    c.show(show_ports=True)
