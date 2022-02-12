Plugins
===================================

-  Mode solver. Frequency Domain solver to compute modes for different
   waveguide geometries. Single frequency/wavelength solving.

   -  MPB. See ``gdsfactory.simulation.modes``

-  FDTD: Finite difference time domain wave propagation. Provides
   Broadband spectrum for a component. Described as Sparameters (ratio
   of output/input field for each port)

   -  lumerical Ansys FDTD. See ``gdsfactory.simulation.lumerical``
   -  meep. See ``gdsfactory.simulation.gmeep``
   -  tidy3d. See ``gdsfactory.simulation.gtidy``

-  Linear circuit solver

   -  `simphony (open
      source) <https://simphonyphotonics.readthedocs.io/en/latest/>`__
   -  `sax (open
      source) <https://sax.readthedocs.io/en/latest/index.html>`__



.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   notebooks/plugins/mpb/001_mpb_waveguide.ipynb
   notebooks/plugins/lumerical/1_fdtd_sparameters.ipynb
   notebooks/plugins/meep/001_meep_sparameters.ipynb
   notebooks/plugins/tidy3d/00_tidy3d.ipynb
   notebooks/plugins/sax/sax.ipynb
