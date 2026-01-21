"""Tests that we get a suitable electrical netlist that represents the physical geometry of our circuit.

We use the `get_missing_models` function in `sax` to extract that it is representing our netlist component correctly.
"""

import gdsfactory as gf
from gdsfactory.get_netlist import SmartPortMatcher, get_netlist_recursive, legacy_namer


class ExcludeOpticalMatcher(SmartPortMatcher):
    def __call__(self, port1: gf.Port, port2: gf.Port) -> bool:
        if port1.port_type == "optical" or port2.port_type == "optical":
            return False
        return super().__call__(port1, port2)


def test_no_effect_on_original_components() -> None:
    passive_mzi = gf.components.mzi2x2_2x2()
    passive_mzi_phase_shifter_netlist_electrical = get_netlist_recursive(
        passive_mzi,
        port_matcher=ExcludeOpticalMatcher(),
    )
    assert passive_mzi_phase_shifter_netlist_electrical is not None


@gf.cell
def hcomponent_top() -> gf.Component:
    c = gf.Component()
    c << hcomponent_l2()
    return c


@gf.cell
def hcomponent_l2() -> gf.Component:
    c = gf.Component()
    c << hcomponent_l3()
    c << hcomponent_l3()
    return c


@gf.cell
def hcomponent_l3() -> gf.Component:
    c = gf.Component()
    return c


def test_n_netlists() -> None:
    c = hcomponent_top()
    netlists = get_netlist_recursive(
        c,
        netlist_namer=legacy_namer,
        instance_namer=legacy_namer,
    )
    # only netlists with hierarchy should be reported
    assert len(netlists) == 2
    assert "hcomponent_top" in netlists
    assert "hcomponent_l2" in netlists
