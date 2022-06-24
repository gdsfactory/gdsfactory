####################
API
####################

.. currentmodule:: gdsfactory

*********************
Geometry Construction
*********************

Classes and functions for construction and manipulation of geometric objects.

.. rubric:: Component definition

.. autosummary::
   :toctree: _autosummary/

   Component
   read.from_yaml
   read.from_np
   read.import_gds

.. currentmodule:: gdsfactory.cross_section

.. rubric:: cross_section functions

.. autosummary::
   :toctree: _autosummary/

   cross_section
   strip
   strip_auto_widen
   heater_metal
   pin
   pn
   strip_heater_metal_undercut
   strip_heater_metal
   strip_heater_doped
   rib_heater_doped
   rib_heater_doped_via_stack

.. currentmodule:: gdsfactory.path

.. rubric:: path functions

.. autosummary::
   :toctree: _autosummary/

   straight
   euler
   arc
   smooth


.. currentmodule:: gdsfactory

.. rubric:: decorators

.. autosummary::
   :toctree: _autosummary/

   cell
   cell_without_validator


.. currentmodule:: gdsfactory.routing

.. rubric:: routing

.. autosummary::
   :toctree: _autosummary/

   add_electrical_pads_shortest
   add_electrical_pads_top
   add_electrical_pads_top_dc
   add_fiber_array
   add_fiber_single
   get_bundle
   get_bundle_from_steps
   get_bundle_from_steps_electrical
   get_bundle_from_steps_electrical_multilayer
   get_bundle_electrical
   get_bundle_electrical_multilayer
   get_bundle_path_length_match
   get_bundle_from_waypoints
   get_bundle_from_waypoints_electrical
   get_bundle_from_waypoints_electrical_multilayer
   get_route
   get_route_electrical
   get_route_electrical_multilayer
   get_routes_bend180
   get_routes_straight
   get_route_sbend
   get_bundle_sbend
   get_route_from_waypoints
   get_route_from_waypoints_electrical
   get_route_from_waypoints_electrical_multilayer
   get_route_from_steps
   get_route_from_steps_electrical
   get_route_from_steps_electrical_multilayer
   fanout2x2
   fanout
   route_ports_to_side
   route_south
   route_quad
   route_sharp
   fanout
