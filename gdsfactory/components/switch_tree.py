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
from gdsfactory.components.mzi_phase_shifter import mzi_phase_shifter
from gdsfactory.components.splitter_tree import splitter_tree

mzi = gf.partial(mzi_phase_shifter, combiner=mmi2x2, delta_length=0)


switch_tree = gf.partial(splitter_tree, coupler=mzi, spacing=(500, 100))


if __name__ == "__main__":
    # c = mzi()
    c = switch_tree(noutputs=16)
    c.show()
