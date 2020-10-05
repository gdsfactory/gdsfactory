# gdsfactory 1.4.0

gdsfactory provides you with generic component functions to build your PDKs and masks for different foundries.

You just need to adapt the functions to your foundry (see pp/samples/pdk) and build your own pdk (see [UBC PDK](https://github.com/gdsfactory/ubc) example).

Gdsfactory extends [phidl](https://github.com/amccaugh/phidl) and [gdspy](https://github.com/heitzmann/gdspy) with some useful photonics functions (see photonics package `pp`) to generate GDS layouts (GDSII is the standard format to create masks sets in the CMOS industry)

- define functions for basic components
- define component sweeps (Design of Experiments or DOEs) in YAML files and GDS masks (together with JSON metadata)
- route optical/electrical ports to pads and grating couplers

## Documentation

- [read online Documentation](https://gdsfactory.readthedocs.io/en/latest/intro.html)
- run pp/samples
- run notebooks
- see latest changes in [CHANGELOG](CHANGELOG.md)

## Installation

If you are on Windows, you need to install a C++ compiler ["Build Tools for Visual Studio"](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)

Once you have `git` and `python3` we recommend to install the latest version from the git repo by copy-pasting this 3 lines into a terminal.

Works for python>=3.6

```
git clone https://github.com/gdsfactory/gdsfactory.git
cd gdsfactory
bash install.sh
```

## Tests

You can run tests with `pytest`. This will run 3 types of tests:

- pytest will test any function in the `pp` package that starts with `test_`
- test_factory: builds all components in the component_type2factory in `pp/components/__init__.py` and checks that the geometric hash is the same
    - any changes in the library need to be approved by running the function `lock_components_with_changes` in `pp/tests/test_factory.py`
- regressions tests: makes sures no unwanted regressions happen. Need to approve changes by running `make test-force` from the repo root directory.
    - `pp/components/test_components.py` stores all the component settings in YAML
    - `pp/components/test_ports.py` stores all port locations in a CSV file


## Modules

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
- notebooks: jupyter-notebooks for training


## `pf` Photonic factory command line interface

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

## Links

- [gdslib](https://github.com/gdsfactory/gdslib): separate repo where we store the component library. Tests ensure the geometric hash of the GDS does not change with the ones locked in the library
  - `component.gds`: GDS
  - `component.json`: JSON file with component properties
  - `component.dat`: FDTD sparameter data
  - `component.ports`: csv with port information
  - `component.nst`: netlist (not implemented yet)
- [ubc PDK](https://github.com/gdsfactory/ubc)
- [awesome photonics list](https://github.com/joamatab/awesome_photonics)
- [gdspy](https://github.com/heitzmann/gdspy)
- [phidl](https://github.com/amccaugh/phidl)
- [picwriter](https://github.com/DerekK88/PICwriter)
