# gdsfactory 2.2.3

gdsfactory provides you with generic component functions to build your PDKs and masks for different foundries.

You just need to adapt the functions to your foundry and build your own PDK (see [UBC PDK](https://github.com/gdsfactory/ubc) example).

Gdsfactory extends [phidl](https://github.com/amccaugh/phidl) and [gdspy](https://github.com/heitzmann/gdspy) with some useful photonics functions (see photonics package `pp`) to generate GDS layouts (GDSII is the standard format to create masks sets in the CMOS industry)

- functions easily adaptable to define components
- define component sweeps (Design of Experiments or DOEs) in YAML files and GDS masks (together with JSON metadata)
- route optical/electrical ports to pads and grating couplers

## Documentation

- [read online Documentation](https://gdsfactory.readthedocs.io/en/latest)
- run pp/samples
- run notebooks
- see latest changes in [CHANGELOG](CHANGELOG.md)

## Installation

Works for python>=3.6.

If you are on Windows, I reccommend you install it with Anaconda3 or Miniconda3.

For Windows, Linux and MacOs you can install the latest released version:

```
conda install -c conda-forge gdspy
pip install gdsfactory
pf install
```

Or you can install the development version:

```
git clone https://github.com/gdsfactory/gdsfactory.git
cd gdsfactory
bash install.sh
```

## Tests

You can run tests with `pytest`. This will run 3 types of tests:

- pytest will test any function in the `pp` package that starts with `test_`
- lytest: writes all components GDS in `run_layouts` and compares them with `ref_layouts`
    - you can check out any changes in the library with `pf diff ref_layouts/bbox.gds run_layouts/bbox.gds`
- regressions tests: avoids unwanted regressions by storing Components ports position and metadata in YAML files. You can force to regenerate those files running `make test-force` from the repo root directory.
    - `pp/test_containers.py` stores container function settings in YAML and port locations in a CSV file
    - `pp/components/test_components.py` stores all the component settings in YAML
    - `pp/components/test_ports.py` stores all port locations in a CSV file


## Modules

- pp photonic-package
  - components: define components
  - drc: check geometry
  - gdsdiff: hash geometry and show differences by displaying boolean operations in klayout
  - klive: stream GDS directly to klayout
  - ports: to connect components
  - routing: add waveguides to connect components
  - samples: python tutorial
  - tests:
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

- [gdsfactory](https://github.com/gdsfactory/gdsfactory): Github repo where we store the gdsfactory code
- [gdslib](https://github.com/gdsfactory/gdslib): separate repo where we store the component library. Tests ensure the geometric hash of the GDS does not change with the ones locked in the library
  - `component.gds`: GDS
  - `component.json`: JSON file with component properties
  - `component.dat`: FDTD sparameter data
  - `component.ports`: CSV with port information
- [ubc PDK](https://github.com/gdsfactory/ubc)
- [awesome photonics list](https://github.com/joamatab/awesome_photonics)
- [gdspy](https://github.com/heitzmann/gdspy)
- [phidl](https://github.com/amccaugh/phidl)
- [picwriter](https://github.com/DerekK88/PICwriter)
