# Mode solvers

A mode solver computes the modes supported by a waveguide cross-section at a particular wavelength. Modes are definite-frequency eigenstates of Maxwell's equations.

You can use 3 open source mode solvers:

1. tidy3d. Finite difference Frequency Domain (FDFD).
2. MPB. FDFD with periodic boundary conditions.
3. Femwell. Finite Element (FEM).

The tidy3d mode solver is also used by the MEOW plugin to get the Sparameters of components via Eigenmode Expansion.
Notice that the tidy3d FDTD solver is not open source as it runs on the cloud server, but the mode solver is open source and runs locally on your computer.

```{tableofcontents}
```
