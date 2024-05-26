"""Tests that we get a suitable electrical netlist that represents the physical geometry of our circuit.

We use the `get_missing_models` function in `sax` to extract that it is representing our netlist component correctly.
"""

import gdsfactory as gf
from gdsfactory.get_netlist import get_netlist_recursive


def test_no_effect_on_original_components():
    passive_mzi = gf.components.mzi2x2_2x2()
    passive_mzi_phase_shifter_netlist_electrical = get_netlist_recursive(
        passive_mzi, exclude_port_types="optical"
    )
    assert passive_mzi_phase_shifter_netlist_electrical is not None
