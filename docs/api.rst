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
   spiral_archimedean
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


*********************
Klayout DRC
*********************

.. currentmodule:: gdsfactory.geometry.write_drc

.. rubric:: klayout drc

.. autosummary::
   :toctree: _autosummary/

   write_drc_deck_macro
   rule_width
   rule_space
   rule_width
   rule_width
   rule_space
   rule_separation
   rule_enclosing


************************
Mode solver Plugins
************************

.. currentmodule:: gdsfactory.simulation.gtidy3d.modes

.. rubric:: Mode solver tidy3d

.. autosummary::
   :toctree: _autosummary/

   Waveguide
   WaveguideCoupler
   sweep_width
   sweep_bend_loss
   group_index
   plot_sweep_width


.. currentmodule:: gdsfactory.simulation.modes

.. rubric:: Mode solver MPB

.. autosummary::
   :toctree: _autosummary/

    waveguide
    coupler
    find_modes_waveguide
    find_modes_coupler
    find_neff_vs_width
    find_mode_dispersion
    find_coupling_vs_gap
    plot_neff_vs_width
    plot_coupling_vs_gap



************************
FDTD Simulation Plugins
************************

.. currentmodule:: gdsfactory.simulation.gmeep

.. rubric:: FDTD meep

.. autosummary::
   :toctree: _autosummary/

   write_sparameters_meep_mpi
   write_sparameters_meep_batch

.. currentmodule:: gdsfactory.simulation.gtidy3d

.. rubric:: FDTD tidy3d

.. autosummary::
   :toctree: _autosummary/

   write_sparameters
   write_sparameters_batch
   write_sparameters_grating_coupler
   write_sparameters_grating_coupler_batch


.. currentmodule:: gdsfactory.simulation.lumerical

.. rubric:: FDTD lumerical

.. autosummary::
   :toctree: _autosummary/

   write_sparameters_lumerical

****************************
Circuit solver Plugins
****************************

.. currentmodule:: gdsfactory.simulation.sax

.. rubric:: SAX

.. autosummary::
   :toctree: _autosummary/

   read
   plot_model
   models


.. currentmodule:: gdsfactory.simulation.simphony

.. rubric:: simphony

.. autosummary::
   :toctree: _autosummary/

    component_to_circuit
    plot_model
    plot_circuit
    plot_circuit_montecarlo
    components


.. currentmodule:: gdsfactory.simulation.lumerical.interconnect

.. rubric:: Lumerical interconnect

.. autosummary::
   :toctree: _autosummary/

    install_design_kit
    add_interconnect_element
    get_interconnect_settings
    send_to_interconnect
    run_wavelength_sweep
    plot_wavelength_sweep
