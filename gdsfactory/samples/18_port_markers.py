"""You can define a function to add pins."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.add_pins import add_pins_triangle


@gf.cell(post_process=[add_pins_triangle])
def straight(**kwargs) -> gf.Component:
    return gf.components.straight(**kwargs)


if __name__ == "__main__":
    c = straight()
    c.show(show_ports=False)
