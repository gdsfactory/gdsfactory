# Plugins

## Component simulation

- Mode solver. Frequency Domain solver to compute modes for different waveguide geometries. Single frequency/wavelength solving. Solve modes supported by a waveguide cross-section
  - MPB. See `gdsfactory.simulation.modes`
- FDTD: Finite difference time domain wave propagation. Provides broadband wavelength spectrum for a component  described as Sparameters (ratio of output/input field for each port) as a function of wavelength.
  - lumerical Ansys FDTD. See `gdsfactory.simulation.lumerical`
  - meep. See `gdsfactory.simulation.gmeep`
  - tidy3d. See `gdsfactory.simulation.gtidy`

FDTD simulators can compute the [Sparameters](https://en.wikipedia.org/wiki/Scattering_parameters) response of a component, which measures the input to output field relationship as a function of wavelength or frequency.

Sparameters are common in RF and photonic simulation.

Frequency circuit simulations solve the Sparameters of a circuit that connects several components, each of which is described by its Sparameters.


## Circuit simulation

- Linear circuit solver
  - [simphony (open source)](https://simphonyphotonics.readthedocs.io/en/latest/)
  - [sax (open source)](https://sax.readthedocs.io/en/latest/index.html)


## Store / cache simulations

Most plugins leverage a file based cache, where you can store simulation results in files to avoid running the same simulation twice.
This saves you a lot of computation time, as each FDTD simulation can take several minutes, or big simulations can take even hours.
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
   fdtd
   notebooks/plugins/sax/sax.ipynb
   notebooks/plugins/simphony/01_components.ipynb
```
