# Plugins

- Mode solver. Frequency Domain solver to compute modes for different waveguide geometries. Single frequency/wavelength solving.
  - MPB. See `gdsfactory.simulation.modes`
- FDTD: Finite difference time domain wave propagation. Provides Broadband spectrum for a component. Described as Sparameters (ratio of output/input field for each port)
  - lumerical Ansys FDTD. See `gdsfactory.simulation.lumerical`
  - meep. See `gdsfactory.simulation.gmeep`
  - tidy3d. See `gdsfactory.simulation.gtidy`
- Linear circuit solver
  - [simphony (open source)](https://simphonyphotonics.readthedocs.io/en/latest/)
  - [sax (open source)](https://sax.readthedocs.io/en/latest/index.html)


FDTD simulators can compute the [Sparameters](https://en.wikipedia.org/wiki/Scattering_parameters) response of a component, which measures the input to output field relationship as a function of wavelength or frequency.

Sparameters are common in RF and photonic simulation.

Frequency circuit simulations solve the Sparameters of a circuit that connects several components, each of which is described by its Sparameters.

Most plugins leverage a file based cache, where you can store simulation results in files to avoid running the same simulation twice.
There are different types of simulation storage that you can use.

- [GIT repo](https://github.com/gdsfactory/gdslib)
- [DVC](https://dvc.org/)
- database


```{eval-rst}
.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   notebooks/plugins/mpb/001_mpb_waveguide.ipynb
   notebooks/plugins/lumerical/1_fdtd_sparameters.ipynb
   notebooks/plugins/meep/001_meep_sparameters.ipynb
   notebooks/plugins/meep/002_gratings.ipynb
   notebooks/plugins/tidy3d/00_tidy3d.ipynb
   notebooks/plugins/sax/sax.ipynb
   notebooks/plugins/simphony/01_components.ipynb
```
