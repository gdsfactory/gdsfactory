# Plugins: Mode solvers

## Component simulation

- Mode solver. Frequency Domain solver to compute modes for different waveguide geometries. Single frequency/wavelength solving. Solve modes supported by a waveguide cross-section
  - MPB. See `gdsfactory.simulation.modes`

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
   :maxdepth: 3
   :titlesonly:
   :caption: Mode solvers:

   notebooks/plugins/mpb/001_mpb_waveguide.ipynb
   notebooks/plugins/tidy3d/01_tidy3d_modes.ipynb
```
