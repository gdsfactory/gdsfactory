"""Write Connectivy checks."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Dict, Layer
from gdsfactory.geometry.write_drc import write_drc_deck_macro

layer_name_to_min_width: Dict[str, float]


class ConnectivyCheck(BaseModel):
    cross_section: CrossSectionSpec
    pin_length: float
    pin_layer: Layer


def write_connectivity_checks(
    connectivity_checks: List[ConnectivyCheck],
) -> str:
    """Return script for photonic port connectivity check. Assumes the photonic port pins are inside the Component.

    Args:
        connectivity_checks: list of connectivity check objects to check for.
    """
    script = ""

    for cc in connectivity_checks:
        xs = gf.get_cross_section(cc.cross_section)
        if not xs.name:
            raise ValueError(f"You need to define {xs}.name")
        script += f"""{xs.name}_pin = input{cc.pin_layer}
{xs.name}_pin = {xs.name}_pin.merged\n
{xs.name}_pin2 = {xs.name}_pin.rectangles.without_area({xs.width} * {2 * cc.pin_length})"""

        if xs.width_wide:
            script += f" - {xs.name}_pin.rectangles.with_area({xs.width_wide} * {2 * cc.pin_length})"

        script += f"""\n{xs.name}_pin2.output(\"port alignment error\")\n
{xs.name}_pin2 = {xs.name}_pin.sized(0.0).merged\n
{xs.name}_pin2.non_rectangles.output(\"port width check\")\n\n"""

    return script


if __name__ == "__main__":
    nm = 1e-3

    connectivity_checks = [
        # ConnectivyCheck(cross_section="strip", pin_length=1 * nm, pin_layer=(1, 10))
        ConnectivyCheck(
            cross_section="strip_auto_widen", pin_length=1 * nm, pin_layer=(1, 10)
        )
    ]
    rules = [write_connectivity_checks(connectivity_checks=connectivity_checks)]

    script = write_drc_deck_macro(rules=rules, layers=None)
    print(script)
