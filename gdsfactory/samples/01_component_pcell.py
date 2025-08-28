"""Based on phidl tutorial.

We'll start by assuming we have a function straight() which already
exists and makes us a simple straight waveguide. Many functions like
this exist in the gdsfactory.components library and are ready-for-use.
We write this one out fully just so it's explicitly clear what's
happening

"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import LayerSpec


@gf.cell
def straight_wide(
    length: float = 5.0, width: float = 1.0, layer: LayerSpec = (1, 0)
) -> gf.Component:
    """Returns straight Component.

    Args:
        length: of the straight.
        width: in um.
        layer: layer spec
    """
    wg = gf.Component(name="straight_sample")
    wg.add_polygon([(0, 0), (length, 0), (length, width), (0, width)], layer=layer)
    wg.add_port(
        name="o1", center=(0, width / 2), width=width, orientation=180, layer=layer
    )
    wg.add_port(
        name="o2", center=(length, width / 2), width=width, orientation=0, layer=layer
    )
    return wg


# ==============================================================================
# Create a blank component
# ==============================================================================
# Let's create a new Component ``c`` which will act as a blank canvas (c can be
# thought of as a blank GDS cell with some special features). Note that when we
# make a Component
