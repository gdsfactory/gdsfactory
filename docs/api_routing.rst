Routing API
===================================================================================================

.. currentmodule:: gdsfactory.routing

route_single
-----------------

.. autosummary::
   :toctree: _autosummary/

   route_single
   route_single_electrical
   route_quad
   route_sharp


route_bundle
---------------------------------------------------------------------------------------------------

When you need to route groups of ports together without them crossing each other You can use a bundle/river/bus router.
`route_bundle` is the generic river bundle bus routing function that will call different function depending on
the port orientation. Get bundle acts as a high level entry point. Based on the angle
configurations of the banks of ports, it decides which sub-routine to call:


.. autosummary::
   :toctree: _autosummary/

   route_bundle



route_bundle_all_angle
---------------------------------------------------------------------------------------------------

.. autosummary::
   :toctree: _autosummary/

   route_bundle_all_angle



route_ports_to_side
---------------------------------------------------------------------------------------------------

For now `route_bundle` is not smart enough to decide whether it should call `route_ports_to_side`.
So you either need to connect your ports to face in one direction first, or to
use `route_ports_to_side` before calling `route_bundle`

.. autosummary::
   :toctree: _autosummary/

   route_ports_to_side
   route_south


fanout
-----------------------------

.. autosummary::
   :toctree: _autosummary/

   fanout2x2


add_fiber_array
---------------------------------------------------------------------------------------------------

In cases where individual components have to be tested, you can generate the array of optical I/O and connect them to the component.

You can connect the waveguides to a 127um pitch fiber array or to individual fibers for input and output.


.. autosummary::
   :toctree: _autosummary/

   add_fiber_array.add_fiber_array


add_pads
-----------------------------


.. autosummary::
   :toctree: _autosummary/

   add_pads_top
   add_pads_bot
   add_electrical_pads_shortest
   add_electrical_pads_top
   add_electrical_pads_top_dc
