# Plugins: Mode solvers

Mode solvers solve modes supported by a waveguide cross-section at a particular wavelength.

You can use 2 mode solvers:

1. MPB (open source).
2. tidy3d (open source). The tidy3d FDTD is not open source. Only the mode solver is open source.

The tidy3d mode solver is also used by the MEOW plugin to get the Sparameters of components via Eigenmode Expansion.

```{eval-rst}
.. toctree::
   :maxdepth: 3
   :titlesonly:
   :caption: Mode solvers:

   notebooks/plugins/fem/01_mode_solving.ipynb
   notebooks/plugins/tidy3d/01_tidy3d_modes.ipynb
   notebooks/plugins/mpb/001_mpb_waveguide.ipynb
   notebooks/plugins/eme/01_meow.ipynb
```
