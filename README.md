# gdsfactory 2.4.3

[![](https://readthedocs.org/projects/gdsfactory/badge/?version=latest)](https://gdsfactory.readthedocs.io/en/latest/?badge=latest)
[![](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![](https://img.shields.io/github/issues/gdsfactory/gdsfactory)](https://github.com/gdsfactory/gdsfactory/issues)
![](https://img.shields.io/github/forks/gdsfactory/gdsfactory)
![](https://img.shields.io/github/stars/gdsfactory/gdsfactory)
[![](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/master/pp)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


![](https://i.imgur.com/v4wpHpg.png)

Gdsfactory is an [EPDA (electronics/photonic design automation)](https://en.wikipedia.org/wiki/Electronic_design_automation) for fabricating Integrated Circuits.

gdsfactory provides you with useful layout functions to build your GDSII components, PDKs and masks for different foundries.

You just need to adapt the functions to your foundry and build your own PDK (see [UBC PDK](https://github.com/gdsfactory/ubc) example).

Gdsfactory extends [phidl](https://github.com/amccaugh/phidl) and [gdspy](https://github.com/heitzmann/gdspy) with some useful photonics functions (see photonics package `pp`) to generate GDS layouts (GDSII is the standard format to create masks sets in the CMOS industry)

- functions easily adaptable to define components
- define component sweeps (Design of Experiments or DOEs) in YAML files and GDS masks (together with JSON metadata)
- route optical/electrical ports to pads and grating couplers

## Documentation

- [read online Documentation](https://gdsfactory.readthedocs.io/en/latest)
- run pp/samples
- run docs/notebooks
- see latest changes in [CHANGELOG](CHANGELOG.md)

gdsfactory is all written in python and requires some basic knowledge of python. If you are new to python you can find many [books](https://jakevdp.github.io/PythonDataScienceHandbook/index.html), [youTube videos](https://www.youtube.com/c/anthonywritescode) and [courses](https://github.com/joamatab/practical-python) available online.


## Installation

Works for python>=3.7 for Windows, MacOs and Linux.
[Github](https://github.com/gdsfactory/gdsfactory/actions) runs all the tests at least once a day for different versions of python (3.7, 3.8, 3.9)

If you are on Windows, I recommend you install it with Anaconda3 or Miniconda3.


You can install [klayout](https://www.klayout.de/) to visualize the GDS files that you create.

For Windows, Linux and MacOs you can install the latest released version:

```
conda install -c conda-forge gdspy
pip install gdsfactory
pf install
```

Or you can install the development version if you want to [contribute](https://gdsfactory.readthedocs.io/en/latest/contribution.html) to gdsfactory:

```
git clone https://github.com/gdsfactory/gdsfactory.git
cd gdsfactory
bash install.sh
```

After installing you should be able to `import pp` from a python script.

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
- docs/notebooks: jupyter-notebooks for training



## Links

- [gdsfactory](https://github.com/gdsfactory/gdsfactory): Github repo where we store the gdsfactory code
- [gdslib](https://github.com/gdsfactory/gdslib): separate package for component circuit models (based on Sparameters).
  - `component.gds`: GDS
  - `component.json`: JSON file with component properties
  - `component.dat`: FDTD Sparameters
  - `component.ports`: CSV with port information
- [ubc PDK](https://github.com/gdsfactory/ubc)
- [awesome photonics list](https://github.com/joamatab/awesome_photonics)
- [gdspy](https://github.com/heitzmann/gdspy)
- [phidl](https://github.com/amccaugh/phidl)
- [picwriter](https://github.com/DerekK88/PICwriter)
