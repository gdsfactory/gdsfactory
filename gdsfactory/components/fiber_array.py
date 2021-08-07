from typing import Tuple

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.fiber import circle


@gf.cell
def fiber_array(
    n: int = 8,
    pitch: float = 127.0,
    core_diameter: float = 10,
    cladding_diameter: float = 125,
    layer_core: Tuple[int, int] = gf.LAYER.WG,
    layer_cladding: Tuple[int, int] = gf.LAYER.WGCLAD,
) -> Component:
    """Returns a fiber array

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

    for i in range(n):
        core = c.add_ref(circle(radius=core_diameter / 2, layer=layer_core))
        cladding = c.add_ref(circle(radius=cladding_diameter / 2, layer=layer_cladding))
        core.movex(i * pitch)
        cladding.movex(i * pitch)
        c.add_port(name=f"F{i}", width=core_diameter, orientation=0)

    return c


if __name__ == "__main__":
    c = fiber_array()
    c.show(show_ports=True)
