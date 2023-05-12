# FDTD

Finite difference time domain (FDTD) simulations directly solve Maxwell’s equations, which are the fundamental description of the physics of classical electrodynamics.

FDTD simulations compute the [Sparameters](https://en.wikipedia.org/wiki/Scattering_parameters) response of a component, which measures the input to output field relationship as a function of wavelength or frequency.

![sparams](https://i.imgur.com/RSOTDIN.png)

gdsfactory provides you a similar python API to drive 3 different FDTD simulators:

  - MEEP
  - tidy3d
  - Lumerical Ansys FDTD

Gdsfactory follows the Sparameters syntax `o1@0,o2@0` where `o1` is the input port `@0` mode 0 (usually fundamental TE mode) and `o2@0` refers to output port `o2` mode 0.

```{tableofcontents}
```
