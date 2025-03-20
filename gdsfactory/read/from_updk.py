"""Read uPDK YAML definition and returns a gdsfactory script.

https://openepda.org/index.html
"""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import yaml

from gdsfactory.serialization import clean_value_name, convert_tuples_to_lists

if TYPE_CHECKING:
    from gdsfactory.typings import LayerSpec, PathType


def from_updk(
    filepath: "PathType",
    filepath_out: "PathType | None" = None,
    layer_bbox: tuple[int, int] = (68, 0),
    layer_bbmetal: tuple[int, int] | None = None,
    layer_label: tuple[int, int] | None = None,
    layer_pin_label: tuple[int, int] | None = None,
    layer_pin: tuple[int, int] | None = None,
    layer_pin_optical: tuple[int, int] | None = None,
    layer_pin_electrical: tuple[int, int] | None = None,
    optical_xsections: Sequence[str] | None = None,
    electrical_xsections: Sequence[str] | None = None,
    layer_text: "LayerSpec | None" = None,
    text_size: float = 2.0,
    activate_pdk: bool = False,
    read_xsections: bool = True,
    use_port_layer: bool = False,
    prefix: str = "",
    suffix: str = "",
    add_plot_to_docstring: bool = True,
    pdk_name: str = "pdk",
) -> str:
    """Read uPDK YAML file and returns a gdsfactory script.

    Args:
        filepath: uPDK filepath.
        filepath_out: optional filepath to save script. if None only returns script and does not save it.
        layer_bbox: layer to draw bounding boxes.
        layer_bbmetal: layer to draw bounding boxes for metal.
        layer_label: layer to draw labels.
        layer_pin_label: layer to draw pin labels.
        layer_pin: layer to draw pins.
        layer_pin_optical: layer to draw optical pins.
        layer_pin_electrical: layer to draw electrical pins.
        optical_xsections: Optional list of names of xsections that will add optical ports.
        electrical_xsections: Optional list of names of xsections that will add electrical ports.
        layer_text: Optional list of layers to add text labels.
        text_size: text size for labels.
        activate_pdk: if True, activate the pdk after writing the script.
        read_xsections: if True, read xsections from uPDK.
        use_port_layer: if True, use the xsection layer for the port.
        prefix: optional prefix to add to the script.
        suffix: optional suffix to add to the script.
        add_plot_to_docstring: if True, add a plot to the docstring.
        pdk_name: name of the pdk.
    """
    optical_xsections = optical_xsections or []
    electrical_xsections = electrical_xsections or []

    filepath = pathlib.Path(filepath)
    filepath = filepath.read_text()
    conf = yaml.safe_load(filepath)
    script = prefix
    script += f"""

import sys
from functools import partial
import gdsfactory as gf
from gdsfactory.get_factories import get_cells
from gdsfactory.add_pins import add_pins_inside2um

cell = gf.cell
layer_bbox = {layer_bbox}
layer_bbmetal = {layer_bbmetal}
layer_pin_label = {layer_pin_label}
layer_pin = {layer_pin}
layer_pin_optical = {layer_pin_optical}
layer_pin_electrical = {layer_pin_electrical}
layer_label = {layer_label}

layer_text = {layer_text or (1, 0)}
text_function = partial(gf.components.text, layer=layer_text, justify="center", size={text_size})

add_pins = partial(add_pins_inside2um, layer_label=layer_label, layer=layer_pin_optical)
"""

    if layer_label:
        script += f"layer_label = {layer_label}\n"

    if read_xsections and "xsections" in conf:
        xsections = conf["xsections"]
        for xsection_name, xsection in xsections.items():
            width = xsection["width"]
            script += f"{xsection_name} = gf.CrossSection(width={width})\n"

        xs = ",".join([f"{name}={name}" for name in xsections.keys()])
        script += "\n"
        script += f"cross_sections = dict({xs})"
        script += "\n"

    for block_name, block in conf["blocks"].items():
        if "parameters" in block:
            parameters = block["parameters"]
        else:
            warnings.warn(f"{block_name=} does not have parameters")
            continue

        parameters_string = (
            ", ".join(
                [
                    f"{clean_value_name(p_name)}:{p['type']}={p['value']}"
                    for p_name, p in parameters.items()
                ]
            )
            if parameters
            else ""
        )

        parameters_doc = (
            "\n    ".join(
                [
                    f"  {p_name}: {p['doc']} (min: {p['min']}, max: {p['max']}, {p['unit']})."
                    if "min" in p
                    else f"  {p_name}: {p['doc']}."
                    for p_name, p in parameters.items()
                ]
            )
            if parameters
            else ""
        )

        parameters_colon = (
            [
                f"{clean_value_name(p_name)}:{{{clean_value_name(p_name)}}}"
                for p_name in parameters
            ]
            if parameters
            else []
        )
        parameters_equal = (
            [
                f"{clean_value_name(p_name)}={{{clean_value_name(p_name)}}}"
                for p_name in parameters
            ]
            if parameters
            else []
        )

        parameters_labels = (
            "\n".join(
                [
                    f"    c.add_label(text=f'{p_name}', position=(xc, yc-{i}/{len(parameters)}/2*ysize), layer=layer_label)\n"
                    for i, p_name in enumerate(parameters_colon)
                ]
            )
            if layer_label and parameters_colon
            else ""
        )
        list_parameters = "\\n".join(f"{p_name}" for p_name in parameters_equal)
        parameters_labels = f"    c.add_label(text=f'Parameters:\\n{list_parameters}', position=(0,0), layer=layer_label)\n"

        docstring = block.get("doc", "")

        plot_docstring = (
            f"""
    .. plot::
      :include-source:

      from {pdk_name} import cells

      c = cells.{block_name}()
      c.draw_ports()
      c.plot()
    """
            if add_plot_to_docstring
            else ""
        )

        if parameters:
            doc = f'"""{docstring}\n\n    Args:\n    {parameters_doc}\n    {plot_docstring}"""'
        else:
            doc = f'"""{docstring}    {plot_docstring}"""'

        cell_name = (
            f"{block_name}:{','.join(parameters_equal)}"
            if parameters_equal
            else block_name
        )

        points = str(block["bbox"]).replace("'", "")
        script += f"""
@gf.cell
def {block_name}({parameters_string})->gf.Component:
    {doc}
    c = gf.Component()
    c.add_polygon({points}, layer=layer_bbox)
    xc = c.x
    yc = c.y
    name = f{cell_name!r}
"""
        if "ysize" in parameters_labels:
            script += """
    ysize = p.ysize
"""
        if layer_bbmetal and "bb_metal" in block:
            for bbmetal in block["bb_metal"].values():
                points = str(bbmetal).replace("'", "")
                script += f"    c.add_polygon({points}, layer=layer_bbmetal)\n"

        script += parameters_labels

        port_layer = "layer" if use_port_layer else "cross_section"

        pins = block.get("pins", {})
        for port_name, port in pins.items():
            port_type = (
                "electrical" if port["xsection"] in electrical_xsections else "optical"
            )

            port_xsection = port["xsection"] if port["xsection"] != "None" else "NONE"
            xya = port["xya"]
            width = port["width"]

            if port_xsection != "None" and not use_port_layer:
                script += f"    c.add_port(name={port_name!r}, {port_layer}={port_xsection!r}, center=({xya[0]}, {xya[1]}), orientation={xya[2]}, port_type={port_type!r})\n"
                script += f"    c.ports[{port_name!r}].info['cross_section'] = {port_xsection!r}\n"
            else:
                script += f"    c.add_port(name={port_name!r}, width={width}, layer={port_xsection!r}, center=({xya[0]}, {xya[1]}), orientation={xya[2]}, port_type={port_type!r})\n"

            if layer_pin_label:
                d = port
                d["name"] = port_name
                d = convert_tuples_to_lists(d)
                text = yaml.dump(d)
                script += f"    c.add_label(text={text!r}, position=({xya[0]}, {xya[1]}), layer=layer_pin_label)\n"
        if layer_text:
            script += "    text = c << text_function(text=name)\n"

            script += "    text.x = xc\n"
            script += "    text.y = yc\n"

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
    c.show()
"""
    if filepath_out:
        dirpath = pathlib.Path(filepath_out).parent
        dirpath.mkdir(parents=True, exist_ok=True)
        filepath_out = pathlib.Path(filepath_out)
        filepath_out.write_text(script)
    return script


if __name__ == "__main__":
    from gdsfactory.config import GDSDIR_TEMP
    from gdsfactory.samples.pdk.fab_c import PDK

    PDK.activate()

    yaml_pdk_decription = PDK.to_updk()
    print(yaml_pdk_decription)
    filepath = GDSDIR_TEMP / "pdk.yaml"
    gdsfactory_script = from_updk(filepath, pdk_name="demo")
    print(gdsfactory_script)
