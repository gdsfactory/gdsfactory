from typing import Tuple

import pp
from pp.component import Component
from pp.components.circle import circle


@pp.cell_with_validator
def fiber(
    core_diameter: float = 10,
    cladding_diameter: float = 125,
    layer_core: Tuple[int, int] = pp.LAYER.WG,
    layer_cladding: Tuple[int, int] = pp.LAYER.WGCLAD,
) -> Component:
    """Returns a fiber."""
    c = Component()
    c.add_ref(circle(radius=core_diameter / 2, layer=layer_core))
    c.add_ref(circle(radius=cladding_diameter / 2, layer=layer_cladding))
    c.add_port(name="F0", width=core_diameter, orientation=0)
    return c


if __name__ == "__main__":
    c = fiber()
    c.show(show_ports=True)
