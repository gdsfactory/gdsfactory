from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.get_netlist_flat import get_netlist_flat


@pytest.mark.skip
def test_flatten_netlist_identical_references():
    """Testing electrical netlist w/ identical component references."""
    # Define compound component
    series_resistors = gf.Component("seriesResistors")
    rseries1 = series_resistors << gf.get_component(
        gf.components.resistance_sheet, width=20, ohms_per_square=20
    )
    rseries2 = series_resistors << gf.get_component(
        gf.components.resistance_sheet, width=20, ohms_per_square=20
    )
    rseries1.connect("pad2", rseries2.ports["pad1"])
    series_resistors.add_port("pad1", port=rseries1.ports["pad1"])
    series_resistors.add_port("pad2", port=rseries2.ports["pad2"])

    # Increase hierarchy levels more
    double_series_resistors = gf.Component("double_seriesResistors")
    rseries1 = double_series_resistors << gf.get_component(series_resistors)
    rseries2 = double_series_resistors << gf.get_component(series_resistors)
    rseries1.connect("pad2", rseries2.ports["pad1"])
    double_series_resistors.add_port("pad1", port=rseries1.ports["pad1"])
    double_series_resistors.add_port("pad2", port=rseries2.ports["pad2"])

    # Define top-level component
    vdiv = gf.Component("voltageDivider")
    r1 = vdiv << double_series_resistors
    r2 = vdiv << series_resistors
    r3 = vdiv << gf.get_component(
        gf.components.resistance_sheet, width=20, ohms_per_square=20
    )
    r4 = vdiv << gf.get_component(
        gf.components.resistance_sheet, width=20, ohms_per_square=20
    )

    r1.connect("pad2", r2.ports["pad1"])
    r3.connect("pad1", r2.ports["pad1"])
    r4.connect("pad1", r3.ports["pad2"])

    vdiv.add_port("gnd1", port=r2.ports["pad2"])
    vdiv.add_port("gnd2", port=r4.ports["pad2"])
    vdiv.add_port("vsig", port=r1.ports["pad1"])
    instances = get_netlist_flat(vdiv, allow_multiple=True)["instances"]
    assert len(instances) == 8, len(instances)


if __name__ == "__main__":
    test_flatten_netlist_identical_references()
    # test_flat_netlist_photonic()
