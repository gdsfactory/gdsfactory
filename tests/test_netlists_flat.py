from __future__ import annotations

import gdsfactory as gf
from gdsfactory.get_netlist_flat import get_netlist_flat


def test_flat_netlist_photonic():
    coupler_lengths = [10, 20, 30, 40]
    coupler_gaps = [0.1, 0.2, 0.4, 0.5]
    delta_lengths = [10, 100, 200]

    c = gf.components.mzi_lattice(
        coupler_lengths=coupler_lengths,
        coupler_gaps=coupler_gaps,
        delta_lengths=delta_lengths,
    )
    gf.get_netlist_flat.get_netlist_flat(c)


def test_flatten_netlist_identical_references():
    """
    Testing electrical netlist w/ identical component references
    """
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
    r3 = (
        vdiv
        << gf.get_component(
            gf.components.resistance_sheet, width=20, ohms_per_square=20
        ).rotate()
    )
    r4 = vdiv << gf.get_component(
        gf.components.resistance_sheet, width=20, ohms_per_square=20
    )

    r1.connect("pad2", r2.ports["pad1"])
    r3.connect("pad1", r2.ports["pad1"], preserve_orientation=True)
    r4.connect("pad1", r3.ports["pad2"], preserve_orientation=True)

    vdiv.add_port("gnd1", port=r2.ports["pad2"])
    vdiv.add_port("gnd2", port=r4.ports["pad2"])
    vdiv.add_port("vsig", port=r1.ports["pad1"])

    assert len(get_netlist_flat(vdiv, allow_multiple=True)["instances"]) == 8


if __name__ == "__main__":
    test_flatten_netlist_identical_references()
    # test_flat_netlist_photonic()
