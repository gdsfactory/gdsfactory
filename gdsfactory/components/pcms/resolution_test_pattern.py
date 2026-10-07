from __future__ import annotations

__all__ = ["resolution_test_pattern"]


import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name(tags=["pcms"])
def resolution_test_pattern(
    radius: float = 50.0,
    n_spokes: int = 36,
    width: float = 1.0,
    layer: LayerSpec = "WG",
) -> Component:
    """Radial Siemens star resolution test pattern.

    Creates a circle of alternating filled/empty pie-shaped wedges.
    The pattern is useful for evaluating lithographic resolution,
    as the feature size decreases toward the center.

    Args:
        radius: Outer radius of the star pattern in um.
        n_spokes: Total number of spokes (filled + empty). Must be even.
        width: Target outer edge width of each spoke in um (informational;
            actual angular width is 360/n_spokes degrees).
        layer: Layer specification for the filled wedges.
    """
    return cf.resolution_test_pattern(
        radius=radius,
        n_spokes=n_spokes,
        width=width,
        layer=layer,
    )


if __name__ == "__main__":
    c = resolution_test_pattern()
    c.show()
