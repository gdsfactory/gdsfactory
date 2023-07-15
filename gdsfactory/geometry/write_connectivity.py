"""Write Connectivy checks."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.geometry.write_drc import write_drc_deck_macro
from gdsfactory.typings import CrossSectionSpec, Dict, Layer, LayerSpec

layer_name_to_min_width: Dict[str, float]

nm = 1e-3


class ConnectivyCheck(BaseModel):
    cross_section: CrossSectionSpec
    pin_length: float
    pin_layer: Layer


def write_connectivity_checks(
    pin_widths: List[float],
    pin_layer: Layer,
    pin_length: float = 1 * nm,
    device_layer: LayerSpec = "DEVREC",
) -> str:
    """Return script for port connectivity check.
    Assumes the port pins are inside the Component.

    Args:
        pin_widths: list of pin widths allowed.
        pin_layer: for the pin markers.
        pin_length: in um.
        device_layer: device recognizion layer.
    """
    device_layer = gf.get_layer(device_layer)
    pin_layer = gf.get_layer(pin_layer)

    script = f"""pin = input{pin_layer}
pin = pin.merged\n
pin2 = pin.rectangles.without_area({pin_widths[0]} * {2 * pin_length})"""

    for w in pin_widths[1:]:
        script += f" - pin.rectangles.with_area({w} * {2 * pin_length})"

    script += """\npin2.output(\"port alignment error\")\n
pin2 = pin.sized(0.0).merged\n
pin2.non_rectangles.output(\"port width check\")\n\n"""

    script += f"""DEVREC = input{device_layer}.raw.merged(2)\n
DEVREC.overlapping(DEVREC).output("Component overlap")\n
    """

    return script


def write_connectivity_checks_per_section(
    connectivity_checks: List[ConnectivyCheck],
    device_layer: LayerSpec = None,
) -> str:
    """Return script for port connectivity check.
    Assumes the port pins are inside the Component and each cross_section has pins on a different layer.
    This is not the recommended way as it only supports two widths per cross_section (cross_section.width and cross_section.width_wide)

    Args:
        connectivity_checks: list of connectivity objects to check for.
        device_layer: device recognizion layer.
    """
    script = ""
    device_layer = gf.get_layer(device_layer)

    for cc in connectivity_checks:
        xs = gf.get_cross_section(cc.cross_section)
        if not xs.name:
            raise ValueError("You need to define cross_section name")
        script += f"""{xs.name}_pin = input{cc.pin_layer}
{xs.name}_pin = {xs.name}_pin.merged\n
{xs.name}_pin2 = {xs.name}_pin.rectangles.without_area({xs.width} * {2 * cc.pin_length})"""

        if xs.width_wide:
            script += f" - {xs.name}_pin.rectangles.with_area({xs.width_wide} * {2 * cc.pin_length})"

        script += f"""\n{xs.name}_pin2.output(\"port alignment error\")\n
{xs.name}_pin2 = {xs.name}_pin.sized(0.0).merged\n
{xs.name}_pin2.non_rectangles.output(\"port width check\")\n\n"""

    if device_layer:
        script += f"""DEVREC = input{device_layer}.raw.merged(2)\n
DEVREC.overlapping(DEVREC).output("Component overlap")\n
    """

    return script


if __name__ == "__main__":
    from gdsfactory.generic_tech import LAYER

    nm = 1e-3

    connectivity_checks = [
        # ConnectivyCheck(cross_section="strip", pin_length=1 * nm, pin_layer=(1, 10))
        ConnectivyCheck(
            cross_section="strip_auto_widen", pin_length=1 * nm, pin_layer=(1, 10)
        )
    ]
    rules = [
        write_connectivity_checks_per_section(connectivity_checks=connectivity_checks),
        "DEVREC",
    ]

    rules = [
        write_connectivity_checks(pin_widths=[0.5, 0.9, 0.45], pin_layer=LAYER.PORT)
    ]
    script = write_drc_deck_macro(rules=rules, layers=None)
    print(script)
