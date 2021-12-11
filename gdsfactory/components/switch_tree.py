"""
                  __
                _|  |_
          __   | |  |_   _
         |  |__| |__|    |
        _|  |__          |dy
         |__|  |  __     |
               |_|  |_   |
                 |  |_   -
                 |__|


           |<-dx->|

"""


import gdsfactory as gf
from gdsfactory.components.mmi2x2 import mmi2x2
from gdsfactory.components.mzi import mzi1x2_2x2
from gdsfactory.components.splitter_tree import splitter_tree
from gdsfactory.components.straight_heater_metal import straight_heater_metal

mzi = gf.partial(
    mzi1x2_2x2,
    combiner=mmi2x2,
    delta_length=0,
    straight_x_top=straight_heater_metal,
    length_x=None,
)

switch_tree = gf.partial(splitter_tree, coupler=mzi, spacing=(500, 100))


if __name__ == "__main__":
    # c = mzi()
    c = switch_tree(noutputs=16)
    c.show()
