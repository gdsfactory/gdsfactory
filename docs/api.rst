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
   ComponentAllAngle
   ComponentReference
   read.import_gds
   read.from_yaml
   read.from_np


.. rubric:: Cell decorators

.. autosummary::
   :toctree: _autosummary/

   cell
   vcell

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
   Section
   cross_section
   strip
   strip_no_ports
   slot
   rib
   rib_bbox
   rib2
   rib_with_trenches
   nitride
   strip_rib_tip
   strip_nitride_tip
   l_with_trenches
   metal1
   metal2
   metal3
   metal_routing
   gs
   gsg
   heater_metal
   npp
   pin
   pn
   pn_with_trenches
   pn_with_trenches_asymmetric
   l_wg_doped_with_trenches
   strip_heater_metal_undercut
   strip_heater_metal
   strip_heater_doped
   rib_heater_doped
   rib_heater_doped_via_stack
   pn_ge_detector_si_contacts
   CrossSectionFactory
   CrossSectionSpec

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
   ComponentSpec
   Layer
   LayerSpec
   LayerSpecs
   MaterialSpec
   PathType
   Step
   MultiCrossSectionAngleSpec

.. currentmodule:: gdsfactory.technology
.. rubric:: technology

.. autosummary::
   :toctree: _autosummary/

   AbstractLayer
   DerivedLayer
   LayerLevel
   LayerMap
   LayerStack
   LayerView
   LayerViews
   LogicalLayer
   lyp_to_dataclass

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
