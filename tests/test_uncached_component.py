from __future__ import annotations

import pytest

import gdsfactory as gf
from gdsfactory.component import UncachedComponentWarning, UncachedComponentError


@gf.cell
def dangerous_intermediate_cells(width=0.5):
    """Example that will show the dangers of using intermediate cells."""
    c = gf.Component("safe")
    c2 = gf.Component(
        "dangerous"
    )  # This should be forbidden as it will create duplicated cells
    c2 << gf.components.hline(width=width)
    c << c2
    return c


@gf.cell
def using_dangerous_intermediate_cells():
    """Example on how things can go wrong.

    Here we try to create to lines with different widths
    they end up with two duplicated cells and a name collision on the intermediate cell
    """
    c = gf.Component()
    c << dangerous_intermediate_cells(width=0.5)
    r3 = c << dangerous_intermediate_cells(width=2)
    r3.movey(5)
    return c


def test_uncached_component_warning():
    """Ensures that an impossible route raises UncachedComponentWarning."""
    c = using_dangerous_intermediate_cells()

    with pytest.warns(UncachedComponentWarning):
        c.write_gds()
    return c


def test_uncached_component_error():
    """Ensures that an impossible route raises UncachedComponentError."""
    c = using_dangerous_intermediate_cells()

    with pytest.raises(UncachedComponentError):
        c.write_gds(on_uncached_component="error")
    return c


if __name__ == "__main__":
    # c = test_uncached_component_warning()
    c = test_uncached_component_error()
    c.show(show_ports=True)
