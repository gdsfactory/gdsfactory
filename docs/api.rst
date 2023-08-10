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
   ComponentReference
   read.import_gds
   read.from_yaml
   read.from_np

.. currentmodule:: gdsfactory.path

.. rubric:: paths

.. autosummary::
   :toctree: _autosummary/

   Path
   straight
   euler
   arc
   spiral_archimedean
   smooth

.. currentmodule:: gdsfactory.cross_section

.. rubric:: cross_section functions

.. autosummary::
   :toctree: _autosummary/

   CrossSection
   Transition
   xsection
   cross_section
   strip
   strip_auto_widen
   heater_metal
   pin
   pn
   pn_with_trenches
   strip_heater_metal_undercut
   strip_heater_metal
   strip_heater_doped
   rib_heater_doped
   rib_heater_doped_via_stack


.. currentmodule:: gdsfactory.path

.. rubric:: transitions

.. autosummary::
   :toctree: _autosummary/

   transition


.. currentmodule:: gdsfactory.geometry

.. rubric:: geometry

.. autosummary::
   :toctree: _autosummary/

   boolean
   boolean_klayout
   boolean_polygons
   fillet
   functions
   invert
   layer_priority
   offset
   outline
   trim
   union
   xor_diff


.. currentmodule:: gdsfactory

.. rubric:: decorators

.. autosummary::
   :toctree: _autosummary/

   cell
   cell_without_validator


.. currentmodule:: gdsfactory.typings
.. rubric:: typings

.. autosummary::
   :toctree: _autosummary/

   Anchor
   CellSpec
   ComponentFactory
   ComponentSpec
   Component
   CrossSection
   CrossSectionFactory
   CrossSectionSpec
   Layer
   LayerSpec
   LayerSpecs
   LayerLevel
   Label
   MaterialSpec
   MultiCrossSectionAngleSpec
   NetlistModel
   PathType
   Route
   RouteFactory
   Routes
   Section
   Step
   StepAllAngle


*********************
Pack
*********************

.. currentmodule:: gdsfactory

.. rubric:: pack

.. autosummary::
   :toctree: _autosummary/

   pack
   grid
   grid_with_text


*********************
Netlist
*********************

.. currentmodule:: gdsfactory.get_netlist

.. rubric:: get_netlist

.. autosummary::
   :toctree: _autosummary/

   get_netlist

.. currentmodule:: gdsfactory.get_netlist_flat

.. autosummary::
   :toctree: _autosummary/

   get_netlist_flat
