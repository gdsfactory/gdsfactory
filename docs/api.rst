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
   read.from_phidl
   read.from_dphox

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
   rule_min_width_or_space
   rule_separation
   rule_enclosing
   rule_area
   rule_density
   rule_not_inside


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

.. currentmodule:: gdsfactory.simulation.fem.mode_solver

.. rubric:: Mode solver Femwell

.. autosummary::
   :toctree: _autosummary/

   compute_cross_section_modes


.. currentmodule:: gdsfactory.simulation.modes

.. rubric:: Mode solver MPB

.. autosummary::
   :toctree: _autosummary/

    find_modes_waveguide
    find_modes_coupler
    find_neff_vs_width
    find_mode_dispersion
    find_coupling_vs_gap
    find_neff_ng_dw_dh
    plot_neff_ng_dw_dh
    plot_neff_vs_width
    plot_coupling_vs_gap

.. currentmodule:: gdsfactory.simulation.eme

.. rubric:: EME

.. autosummary::
   :toctree: _autosummary/

    MEOW


************************
FDTD Simulation Plugins
************************

.. rubric:: common FDTD functions

.. currentmodule:: gdsfactory.simulation.plot

.. autosummary::
   :toctree: _autosummary/

   plot_sparameters
   plot_imbalance2x2
   plot_loss2x2

.. currentmodule:: gdsfactory.simulation

.. autosummary::
   :toctree: _autosummary/

   get_effective_indices


.. currentmodule:: gdsfactory.simulation.gmeep

.. rubric:: FDTD meep

.. autosummary::
   :toctree: _autosummary/

   write_sparameters_meep
   write_sparameters_meep_mpi
   write_sparameters_meep_batch
   write_sparameters_grating
   write_sparameters_grating_mpi
   write_sparameters_grating_batch

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

   read.model_from_csv
   read.model_from_component
   plot_model
   models


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
