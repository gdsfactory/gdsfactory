"""Tests that we get a suitable electrical netlist that represents the physical geometry of our circuit.

We use the `get_missing_models` function in `sax` to extract that it is representing our netlist component correctly.
"""

import gdsfactory as gf
from gdsfactory.export.to_yaml import to_yaml_recursive


def test_no_effect_on_original_components():
    passive_mzi = gf.components.mzi2x2_2x2()
    passive_mzi_phase_shifter_netlist_electrical = to_yaml_recursive(
        passive_mzi, exclude_port_types="optical"
    )
    assert passive_mzi_phase_shifter_netlist_electrical is not None


@gf.cell
def hcomponent_top():
    c = gf.Component()
    c << hcomponent_l2()
    return c


@gf.cell
def hcomponent_l2():
    c = gf.Component()
    c << hcomponent_l3()
    c << hcomponent_l3()
    return c


@gf.cell
def hcomponent_l3():
    c = gf.Component()
    return c


def test_n_netlists():
    c = hcomponent_top()
    netlists = to_yaml_recursive(c)
    # only netlists with hierarchy should be reported
    assert len(netlists) == 2
    assert "hcomponent_top" in netlists
    assert "hcomponent_l2" in netlists
