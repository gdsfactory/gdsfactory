# GDSfactory 1.1.9

gdsfactory provides you with a set of useful generic component templates to build your PDKs and send your masks to the foundry.

You just need to adapt the templates to your foundry (see pp/samples/pdk)

GDSII is the standard format to create masks sets in the CMOS industry.

Gdsfactory extends [phidl](https://github.com/amccaugh/phidl) and [gdspy](https://github.com/heitzmann/gdspy) with some useful photonics functions (see photonics package `pp`)

- define templates for basic components
- define component sweeps (Design of Experiments or DOEs) in YAML files
- route optical/electrical ports to pads and grating couplers

# Documentation

- [read Documentation](https://gdsfactory.readthedocs.io/en/latest/intro.html)
- run gdsfactory/pp/samples
- run gdsfactory/notebooks

# Installation

If you are on Windows, you need to install a C++ compiler ["Build Tools for Visual Studio"](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)

Once you have `git` and `python3` we recommend to install the latest version from the git repo by copy-pasting this 3 lines into a terminal.

```
git clone https://github.com/gdsfactory/gdsfactory.git
cd gdsfactory
bash install.sh
```

# Modules in this repo

- pp photonic-package
  - components: define components
  - drc: check geometry
  - ports: to connect components
  - klive: stream GDS direcly to klayout
  - routing: add waveguides to connect components
  - samples: python tutorial
  - tests:
- gdsdiff: hash geometry and show differences by displaying boolean operations in klayout
- klayout: klayout generic tech layers and klive macro

# Related repos

- notebooks: jupyter-notebooks for training
- gdslib: separate repo where we store the component outputs. Tests ensure the geometric hash of the GDS does not change with the ones locked in the [library](https://github.com/gdslib/gdslib)
  - `component.nst`: netlist
  - `component.dat`: FDTD sparameter data
  - `component.ice`: interconnect
  - `component.md`: report
  - `component.ports`: csv with port information
  - `component.json`: JSON file with component properties
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

- [gdspy](https://github.com/heitzmann/gdspy)
- [phidl](https://github.com/amccaugh/phidl)
- [picwriter](https://github.com/DerekK88/PICwriter)
