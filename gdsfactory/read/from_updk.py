"""Read uPDK YAML definition and returns a gdsfactory script.

https://openepda.org/index.html
"""

from __future__ import annotations

from typing import IO, Tuple, Optional, List
import io
import pathlib
from omegaconf import OmegaConf

from gdsfactory.typings import PathType


def from_updk(
    filepath: PathType,
    filepath_out: Optional[PathType] = None,
    layer_bbox: Tuple[int, int] = (68, 0),
    layer_label: Optional[Tuple[int, int]] = None,
    optical_xsections: Optional[List[str]] = None,
    electrical_xsections: Optional[List[str]] = None,
) -> str:
    """Read uPDK definition and returns a gdsfactory script.

    Args:
        filepath: uPDK filepath definition.
        filepath_out: optional filepath to save script. if None only returns script and does not save it.
        layer_bbox: layer to draw bounding boxes.
        optical_xsections: Optional list of names of xsections that will add optical ports.
        electrical_xsections: Optional list of names of xsections that will add electrical ports.
    """

    optical_xsections = optical_xsections or []
    electrical_xsections = electrical_xsections or []

    if isinstance(filepath, (str, pathlib.Path, IO)):
        filepath = (
            io.StringIO(filepath)
            if isinstance(filepath, str) and "\n" in filepath
            else filepath
        )

        conf = OmegaConf.load(
            filepath
        )  # nicer loader than conf = yaml.safe_load(filepath)
    else:
        conf = OmegaConf.create(filepath)

    script = f"""
import sys
import gdsfactory as gf
from gdsfactory.get_factories import get_cells

layer_bbox = {layer_bbox}
"""

    if layer_label:
        script += f"layer_label = {layer_label}\n"

    for xsection_name, xsection in conf.xsections.items():
        script += f"{xsection_name} = gf.CrossSection(width={xsection.width})\n"

    xs = ",".join([f"{name}={name}" for name in conf.xsections.keys()])
    script += "\n"
    script += f"cross_sections = dict({xs})"
    script += "\n"

    for block_name, block in conf.blocks.items():
        parameters = block.parameters
        parameters_string = (
            ", ".join(
                [f"{p_name}:{p.type}={p.value}" for p_name, p in parameters.items()]
            )
            if parameters
            else ""
        )
        parameters_doc = (
            "\n    ".join(
                [
                    f"  {p_name}: {p.doc} (min: {p.min}, max: {p.max}, {p.unit})."
                    for p_name, p in parameters.items()
                ]
            )
            if parameters
            else ""
        )

        parameters_labels = (
            "\n".join(
                [
                    f"    c.add_label(text=f'{p_name}:{{{p_name}}}', position=(xc, yc-{i}/{len(parameters)}*ysize/2), layer=layer_label)"
                    for i, p_name in enumerate(parameters)
                ]
            )
            if layer_label and parameters
            else ""
        )

        if parameters:
            doc = f'"""{block.doc}\n\n    Args:\n    {parameters_doc}\n    """'
        else:
            doc = f'"""{block.doc}"""'

        points = str(block.bbox).replace("'", "")
        script += f"""
@gf.cell
def {block_name}({parameters_string})->gf.Component:
    {doc}
    c = gf.Component()
    p = c.add_polygon({points}, layer=layer_bbox)
    xc, yc = p.center
    ysize = p.ysize
"""
        script += parameters_labels

        for port_name, port in block.pins.items():
            port_type = (
                "electrical" if port.xsection in electrical_xsections else "optical"
            )
            script += f"""
    c.add_port(name={port_name!r}, width={port.width}, cross_section={port.xsection!r}, center=({port.xya[0]}, {port.xya[1]}), orientation={port.xya[2]}, port_type={port_type!r})"""

        script += """
    return c
"""

    script += f"""
cells = get_cells(sys.modules[__name__])
pdk = gf.Pdk(name={conf.header.description!r}, cells=cells, cross_sections=cross_sections)
pdk.activate()

if __name__ == "__main__":
    c = {block_name}()
    c.show(show_ports=True)
"""
    if filepath_out:
        filepath_out = pathlib.Path(filepath_out)
        filepath_out.write_text(script)
    return script
