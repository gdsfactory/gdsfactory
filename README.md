# GDS factory 1.1.1

Python package to generate GDS layouts.

GDSII is the standard format to create masks sets in the CMOS industry.

This package adds some extra functionalities to [phidl](https://github.com/amccaugh/phidl):

- define components by netlist
- define component sweeps (Design of Experiments) in YAML files
- define templates for basic components
- route optical/electrical ports

# Documentation

[read Documentation](https://github.com/PsiQ/gdsfactory)

# Installation

If you are on Windows, you need to install a C++ compiler ["Build Tools for Visual Studio"](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)

Once you have `git` and `python3` we recommend to install the latest version from the gitlab repo by copy-pasting this 3 lines into a terminal.

```
make install
```

# Submodules in this repo

- pp photonic-package
  - components: define components
  - drc: check geometry
  - ports: to connect components
  - routing: add waveguides to connect components
  - samples: python tutorial
  - tests
- gdsdiff: hash geometry and show differences by displaying boolean operations in klayout
- klive: stream GDS direcly to klayout

# Related repos

- notebooks: jupyter-notebooks for training
- gdslib: separate repo where we store the component outputs
  - `component.nst`: netlist
  - `component.dat`: FDTD sparameter data
  - `component.ice`: interconnect
  - `component.md`: report
  - `component.ports`: csv with port information
  - `component.properties`: JSON file with component properties
  - `component.gds`: GDS

# `pf` Photonic factory command line interface

`pf` builds, tests, and configures masks and components from the command line. Just type `pf` in a terminal.

```
Commands:
  build    Commands for building masks
  config   Work with pdk.CONFIG
  drc      Run DRC
  log      Work with logs
  show     Show a GDS file in Klayout using KLive
  status   Shows version and configuration info
  test     Run tests using pytest.
```

# Acknowledgements

- [phidl](https://github.com/amccaugh/phidl)
- [picwriter](https://github.com/DerekK88/PICwriter)
