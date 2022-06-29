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
   read.from_yaml
   read.from_np
   read.import_gds

.. currentmodule:: gdsfactory.path

.. rubric:: paths

.. autosummary::
   :toctree: _autosummary/

   Path
   straight
   euler
   arc
   smooth

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

.. rubric:: transitions

.. autosummary::
   :toctree: _autosummary/

   transition


.. currentmodule:: gdsfactory

.. rubric:: decorators

.. autosummary::
   :toctree: _autosummary/

   cell
   cell_without_validator


.. currentmodule:: gdsfactory.types
.. rubric:: types

.. autosummary::
   :toctree: _autosummary/

   Layer
   ComponentSpec
   LayerSpec
   CrossSectionSpec
   CellSpec
