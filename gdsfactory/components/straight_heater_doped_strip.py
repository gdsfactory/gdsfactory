r"""
top view

.. code::

                          length
      <-|--------|--------------------------------->
        |        | length_section
        |<--------------------------->
       length_via_stack
        |<------>|
        |________|_____________________________
       /|        |____________________|        |
      / |viastack|                    |via_stack |
      \ | size   |____________________|        |
       \|________|____________________|________|
                                      |        |
                  cross_section_heater|        |
                                      |        |
                                      |        |
                                      |________|

cross_section

.. code::

                              |<------width------>|
      ____________             ___________________               ______________
     |            |           |     undoped Si    |             |              |
     |layer_heater|           |  intrinsic region |<----------->| layer_heater |
     |____________|           |___________________|             |______________|
                                                                 <------------>
                                                    heater_gap     heater_width

"""


from functools import partial

from gdsfactory.components.straight_heater_doped_rib import straight_heater_doped_rib
from gdsfactory.components.via_stack import via_stack_npp_m1
from gdsfactory.cross_section import strip_heater_doped

straight_heater_doped_strip = partial(
    straight_heater_doped_rib,
    cross_section_heater=strip_heater_doped,
    via_stack=via_stack_npp_m1,
)


if __name__ == "__main__":
    # c = straight_heater_doped_strip(length=100)
    # c = test_straight_heater_doped_strip_ports()
    c = straight_heater_doped_strip()
    c.show()
