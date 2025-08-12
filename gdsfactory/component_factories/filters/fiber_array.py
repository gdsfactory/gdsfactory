from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.shapes import circle
from gdsfactory.typings import LayerSpec


def fiber_array_factory(
    n: int = 8,
    pitch: float = 127.0,
    core_diameter: float = 10,
    cladding_diameter: float = 125,
    layer_core: LayerSpec = "WG",
    layer_cladding: LayerSpec = "WGCLAD",
) -> Component:
    """Returns a fiber array.

    Args:
        n: number of fibers.
        pitch: spacing.
        core_diameter: 10um.
        cladding_diameter: in um.
        layer_core: layer spec for fiber core.
        layer_cladding: layer spec for fiber cladding.

    .. code::

        pitch
         <->
        _________
       |         | lid
       | o o o o |
       |         | base
       |_________|
          length
    """
    c = Component()
    layer_core = gf.get_layer(layer_core)

    # Reuse circle components to avoid redundant creation
    core_circle = circle(radius=core_diameter / 2, layer=layer_core)
    cladding_circle = circle(radius=cladding_diameter / 2, layer=layer_cladding)

    for i in range(n):
        core = c.add_ref(core_circle)
        cladding = c.add_ref(cladding_circle)
        core.movex(i * pitch)
        cladding.movex(i * pitch)
        c.add_port(
            name=f"F{i}",
            width=core_diameter,
            orientation=0,
            layer=layer_core,
            center=(i * pitch, 0),
        )

    return c
