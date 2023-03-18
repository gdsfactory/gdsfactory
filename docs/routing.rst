Routing API
===================================================================================================

.. currentmodule:: gdsfactory.routing

single route
-----------------

.. autosummary::
   :toctree: _autosummary/

   get_route
   get_route_from_steps
   route_quad
   route_sharp


bundle route
---------------------------------------------------------------------------------------------------

When you need to route groups of ports together without them crossing each other You can use a bundle/river/bus router.
`get bundle` is the generic river bundle bus routing function that will call different function depending on
the port orientation. Get bundle acts as a high level entry point. Based on the angle
configurations of the banks of ports, it decides which sub-routine to call:

 - `get_bundle_same_axis`, banks or ports facing each other (but with arbitrary and varying pitch
   on each side)
 - `get_bundle_corner`, banks of ports with 90Deg / 270Deg between them (again pitch is flexible on
   both sides)
 - `get_bundle_udirect`, banks of ports with direct U-turns
 - `get_bundle_uindirect`, banks of ports with indirect U-turns

.. autosummary::
   :toctree: _autosummary/

   get_bundle



get_bundle_from_steps
---------------------------------------------------------------------------------------------------

.. autosummary::
   :toctree: _autosummary/

   get_bundle_from_steps
   get_bundle_path_length_match


get_bundle_all_angle
---------------------------------------------------------------------------------------------------

.. autosummary::
   :toctree: _autosummary/

   get_bundle_all_angle



route_ports_to_side
---------------------------------------------------------------------------------------------------

For now `get_bundle` is not smart enough to decide whether it should call `route_ports_to_side`.
So you either need to connect your ports to face in one direction first, or to
use `route_ports_to_side` before calling `get_bundle`

.. autosummary::
   :toctree: _autosummary/

   route_ports_to_side
   route_south


fanout
-----------------------------

.. autosummary::
   :toctree: _autosummary/

   get_routes_bend180
   get_routes_straight
   get_route_sbend
   get_bundle_sbend
   fanout2x2
   fanout_component
   fanout_ports


add_fiber_array / add_fiber_single
---------------------------------------------------------------------------------------------------

In cases where individual components have to be tested, you can generate the array of optical I/O and connect them to the component.

You can connect the waveguides to a 127um pitch fiber array or to individual fibers for input and output.


.. autosummary::
   :toctree: _autosummary/

   add_fiber_array.add_fiber_array
   add_fiber_single.add_fiber_single


add_pads
-----------------------------


.. autosummary::
   :toctree: _autosummary/

   add_pads_top
   add_pads_bot
   add_electrical_pads_shortest
   add_electrical_pads_top
   add_electrical_pads_top_dc
