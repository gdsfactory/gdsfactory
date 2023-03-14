from __future__ import annotations

import pytest
import warnings

import gdsfactory as gf
from gdsfactory.component import (
    UncachedComponentWarning,
    UncachedComponentError,
    Component,
)


@gf.cell
def dangerous_intermediate_cells(width=0.5) -> Component:
    """Example that will show the dangers of using intermediate cells."""
    c = gf.Component("safe")
    c2 = gf.Component(
        "dangerous"
    )  # This should be forbidden as it will create duplicated cells
    c2 << gf.components.hline(width=width)
    c << c2
    return c


@gf.cell
def using_dangerous_intermediate_cells() -> Component:
    """Example on how things can go wrong.

    Here we try to create to lines with different widths
    they end up with two duplicated cells and a name collision on the intermediate cell
    """
    c = gf.Component()
    c << dangerous_intermediate_cells(width=0.5)
    r3 = c << dangerous_intermediate_cells(width=2)
    r3.movey(5)
    return c


def test_uncached_component_warning() -> Component:
    """Ensures that an UncachedComponentWarning is raised by default when a GDS with uncached cells is written"""
    c = using_dangerous_intermediate_cells()

    with pytest.warns(UncachedComponentWarning):
        c.write_gds()
    return c


def test_uncached_component_ignore() -> Component:
    """Ensures that no warnings are raised when a GDS with uncached cells is written and on_uncached_component="ignore"."""
    c = using_dangerous_intermediate_cells()

    with warnings.catch_warnings():
        # throw an error and fail the test of an UncachedComponentWarning is thrown
        warnings.filterwarnings("error", category=UncachedComponentWarning)
        c.write_gds(on_uncached_component="ignore")
    return c


def test_show_does_not_warn() -> Component:
    """Ensures that no warnings are raised when a GDS with uncached cells is written and on_uncached_component="ignore"."""
    c = using_dangerous_intermediate_cells()

    with warnings.catch_warnings():
        # throw an error and fail the test of an UncachedComponentWarning is thrown
        warnings.filterwarnings("error", category=UncachedComponentWarning)
    return c


def test_uncached_component_error() -> Component:
    """Ensures that an UncachedComponentError is raised when a GDS with uncached cells is written and on_uncached_component="error"."""
    c = using_dangerous_intermediate_cells()

    with pytest.raises(UncachedComponentError):
        c.write_gds(on_uncached_component="error")
    return c


if __name__ == "__main__":
    # c = test_uncached_component_warning()
    c = test_uncached_component_error()
    c.show(show_ports=True)
