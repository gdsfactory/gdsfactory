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
   cross_section
   strip
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


.. currentmodule:: gdsfactory

.. rubric:: boolean

.. autosummary::
   :toctree: _autosummary/

   boolean


.. currentmodule:: gdsfactory

.. rubric:: decorators

.. autosummary::
   :toctree: _autosummary/

   cell


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
   MaterialSpec
   MultiCrossSectionAngleSpec
   PathType
   Section
   Step


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
