"""Read uPDK YAML definition and returns a gdsfactory script.

https://openepda.org/index.html
"""

from __future__ import annotations

import io
import pathlib
from typing import IO

from omegaconf import OmegaConf

from gdsfactory.typings import LayerSpec, PathType


def from_updk(
    filepath: PathType,
    filepath_out: PathType | None = None,
    layer_bbox: tuple[int, int] = (68, 0),
    layer_bbmetal: tuple[int, int] | None = None,
    layer_label: tuple[int, int] | None = None,
    layer_pin_label: tuple[int, int] | None = None,
    layer_pin: tuple[int, int] | None = None,
    layer_pin_optical: tuple[int, int] | None = None,
    layer_pin_electrical: tuple[int, int] | None = None,
    optical_xsections: list[str] | None = None,
    electrical_xsections: list[str] | None = None,
    layer_text: LayerSpec | None = None,
    text_size: float = 2.0,
    activate_pdk: bool = False,
    read_xsections: bool = True,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Read uPDK definition and returns a gdsfactory script.

    Args:
        filepath: uPDK filepath definition.
        filepath_out: optional filepath to save script. if None only returns script and does not save it.
        layer_bbox: layer to draw bounding boxes.
        layer_bbmetal: layer to draw bounding boxes for metal.
        optical_xsections: Optional list of names of xsections that will add optical ports.
        electrical_xsections: Optional list of names of xsections that will add electrical ports.
        layers_text: Optional list of layers to add text labels.
        text_size: text size for labels.
        activate_pdk: if True, activate the pdk after writing the script.
        read_xsections: if True, read xsections from uPDK.
        prefix: optional prefix to add to the script.
        suffix: optional suffix to add to the script.
    """

    optical_xsections = optical_xsections or []
    electrical_xsections = electrical_xsections or []

    if isinstance(filepath, str | pathlib.Path | IO):
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

    script = prefix
    script += f"""

import sys
import gdsfactory as gf
from gdsfactory.get_factories import get_cells
from gdsfactory.add_pins import add_pins_inside2um

cell = gf.partial(gf.cell, naming_style='updk', autoname=False)
layer_bbox = {layer_bbox}
layer_bbmetal = {layer_bbmetal}
layer_pin_label = {layer_pin_label}
layer_pin = {layer_pin}
layer_pin_optical = {layer_pin_optical}
layer_pin_electrical = {layer_pin_electrical}
layer_label = {layer_label}

layer_text = {layer_text or (1, 0)}
text_function = gf.partial(gf.components.text, layer=layer_text, justify="center", size={text_size})

add_pins = partial(add_pins_inside2um, layer_label=layer_label, layer=layer_pin_optical)
"""

    if layer_label:
        script += f"layer_label = {layer_label}\n"

    if read_xsections and "xsections" in conf:
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
                    if hasattr(p, "min")
                ]
            )
            if parameters
            else ""
        )

        parameters_colon = (
            [f"{p_name}:{{{p_name}}}" for p_name in parameters] if parameters else []
        )
        parameters_equal = (
            [f"{p_name}={{{p_name}}}" for p_name in parameters] if parameters else []
        )

        parameters_labels = (
            "\n".join(
                [
                    "    c.add_label(text=f'{}', position=(xc, yc-{}/{}/2*ysize), layer=layer_label)\n".format(
                        p_name, i, len(parameters)
                    )
                    for i, p_name in enumerate(parameters_colon)
                ]
            )
            if layer_label and parameters_colon
            else ""
        )

        if parameters:
            doc = f'"""{block.doc}\n\n    Args:\n    {parameters_doc}\n    """'
        else:
            doc = f'"""{block.doc}"""'

        cell_name = (
            f"{block_name}:{','.join(parameters_equal)}"
            if parameters_equal
            else block_name
        )

        points = str(block.bbox).replace("'", "")
        script += f"""
@cell
def {block_name}({parameters_string})->gf.Component:
    {doc}
    c = gf.Component()
    p = c.add_polygon({points}, layer=layer_bbox)
    xc, yc = p.center
    ysize = p.ysize
    name = f{cell_name!r}
"""
        if layer_bbmetal and "bb_metal" in block:
            for bbmetal in block["bb_metal"].values():
                points = str(bbmetal).replace("'", "")
                script += f"    c.add_polygon({points}, layer=layer_bbmetal)\n"

        script += parameters_labels

        for port_name, port in block.pins.items():
            port_type = (
                "electrical" if port.xsection in electrical_xsections else "optical"
            )

            if port.xsection != "None":
                script += f"    c.add_port(name={port_name!r}, width={port.width}, cross_section={port.xsection!r}, center=({port.xya[0]}, {port.xya[1]}), orientation={port.xya[2]}, port_type={port_type!r})\n"
            else:
                script += f"    c.add_port(name={port_name!r}, width={port.width}, layer=(0, 0), center=({port.xya[0]}, {port.xya[1]}), orientation={port.xya[2]}, port_type={port_type!r})\n"

            if layer_pin_label:
                d = OmegaConf.to_container(port)
                d["name"] = port_name
                text = OmegaConf.to_yaml(d)
                script += f"    c.add_label(text={text!r}, position=({port.xya[0]}, {port.xya[1]}), layer=layer_pin_label)\n"
        if layer_text:
            script += "    text = c << text_function(text=name)\n"

            script += "    text.center = (xc, yc)\n"
            script += "    c.absorb(text)\n"

        script += """
    c.name = name
    if layer_pin:
        add_pins(c, layer=layer_pin)
    return c
"""

    if activate_pdk:
        script += f"""
cells = get_cells(sys.modules[__name__])
pdk = gf.Pdk(name={conf.header.description!r}, cells=cells, cross_sections=cross_sections)
pdk.activate()
"""

    script += f"""
{suffix}

if __name__ == "__main__":
    c = {block_name}()
    c.show(show_ports=True)
"""
    if filepath_out:
        dirpath = filepath_out.parent
        dirpath.mkdir(parents=True, exist_ok=True)
        filepath_out = pathlib.Path(filepath_out)
        filepath_out.write_text(script)
    return script


if __name__ == "__main__":
    from gdsfactory.samples.pdk.fab_c import pdk

    yaml_pdk_decription = pdk.to_updk()
    gdsfactory_script = from_updk(yaml_pdk_decription)
