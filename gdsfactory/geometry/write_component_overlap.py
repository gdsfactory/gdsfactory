"""Write Connectivy checks."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.geometry.write_drc import write_drc_deck_macro
from gdsfactory.typings import LayerSpec


def write_component_overlap(DEVREC_layer: LayerSpec = "DEVREC") -> str:
    """Return script for component overlap.

    Args:
        DEVREC_layer: layer for component recognition.

    Returns:
        drc script.
    """
    DEVREC_layer = gf.get_layer(DEVREC_layer)

    return f"""DEVREC = input{DEVREC_layer}\n
DEVREC2 = DEVREC.dup()\n
DEVREC.overlapping(DEVREC).output("Component overlap")\n
    """


if __name__ == "__main__":
    script_ = write_component_overlap()

    script = write_drc_deck_macro(rules=[script_])
    
    print(script)
