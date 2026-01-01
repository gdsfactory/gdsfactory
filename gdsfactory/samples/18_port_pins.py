"""You can define a function to add pins."""

import gdsfactory as gf
from gdsfactory.gpdk import PDK

PDK.activate()

if __name__ == "__main__":
    c = gf.components.straight()
    c.draw_ports()
    c.show()
