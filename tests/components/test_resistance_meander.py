"""Tests for resistance_meander."""

import gdsfactory as gf


def test_resistance_meander_two_different_meanders() -> None:
    """Two meanders with different settings must not clash on cell names.

    Regression test for #3042: the meander wire was built as a Component named
    "net", so building a second meander raised "Cellname net already exists in
    the layout/KCLayout".
    """
    c = gf.Component()
    meander1 = c << gf.components.resistance_meander(num_squares=1000)
    meander2 = c << gf.components.resistance_meander(num_squares=500)
    assert meander1.cell.name != meander2.cell.name
