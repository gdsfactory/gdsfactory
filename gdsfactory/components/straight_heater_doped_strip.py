r"""
top view

.. code::

                          length
      <-|--------|--------------------------------->
        |        | length_section
        |<--------------------------->
       length_contact
        |<------>|
        |________|_____________________________
       /|        |____________________|        |
      / |viastack|                    |contact |
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


import gdsfactory as gf
from gdsfactory.components.straight_heater_doped_rib import straight_heater_doped_rib
from gdsfactory.components.via_stack import via_stack_npp
from gdsfactory.cross_section import strip_heater_doped

straight_heater_doped_strip = gf.partial(
    straight_heater_doped_rib,
    cross_section_heater=strip_heater_doped,
    via_stack_contact=via_stack_npp,
)


if __name__ == "__main__":
    # c = straight_heater_doped_strip(length=100)
    # c = test_straight_heater_doped_strip_ports()
    c = straight_heater_doped_strip()
    c.show()
