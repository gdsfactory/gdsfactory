# gdsfactory 3.0.0

[![](https://readthedocs.org/projects/gdsfactory/badge/?version=latest)](https://gdsfactory.readthedocs.io/en/latest/?badge=latest)
[![](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![](https://img.shields.io/github/issues/gdsfactory/gdsfactory)](https://github.com/gdsfactory/gdsfactory/issues)
![](https://img.shields.io/github/forks/gdsfactory/gdsfactory)
![](https://img.shields.io/github/stars/gdsfactory/gdsfactory)
[![](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/master/gdsfactory)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![](https://i.imgur.com/v4wpHpg.png)

gdsfactory is an [EDA (electronics design automation)](https://en.wikipedia.org/wiki/Electronic_design_automation) tool to Layout Integrated Circuits.

It is based on [phidl](https://github.com/amccaugh/phidl) and [gdspy](https://github.com/heitzmann/gdspy) and provides you with functions to build your GDSII components, PDKs and masks for different foundries.

You just need to adapt the functions to your foundry and build your own library of elements (see [UBC PDK](https://github.com/gdsfactory/ubc) example).

gdsfactory provides you with:

- functions easily adaptable to define components
- functions to route electrical ports to pads and optical ports grating couplers
- functions to define components, circuit netlists or masks in YAML files

As inputs, gdsfactory needs you to write python functions, python dataclasses or YAML files to describe your layouts.

As output it creates GDSII files ([GDSII](https://en.wikipedia.org/wiki/GDSII) is the standard format to describe masks sets in the CMOS industry).
It also can output JSON files with components settings (that you can use for measurement and data analysis) and JSON or CSV files for testing the devices after fabrication. You can also easily adapt this metadata output files to your needs.

## Documentation

- [read online Documentation](https://gdsfactory.readthedocs.io/en/latest)
- run gdsfactory/samples
- run docs/notebooks

gdsfactory is all written in python and requires some basic knowledge of python. If you are new to python you can find many [books](https://jakevdp.github.io/PythonDataScienceHandbook/index.html), [youTube videos](https://www.youtube.com/c/anthonywritescode) and [courses](https://github.com/joamatab/practical-python) available online.

## Installation

First, you need to install [klayout](https://www.klayout.de/) to visualize the GDS files that you create.

gdsfactory works for python>=3.7 in Windows, MacOs and Linux.
[Github](https://github.com/gdsfactory/gdsfactory/actions) runs all the tests at least once a day for different versions of python (3.7, 3.8, 3.9)

If you are on Windows, I recommend you install gdsfactory with Anaconda3 or Miniconda3.

```
conda install -c conda-forge gdspy
pip install gdsfactory
gf tool install
```

For Linux and MacOs you can also install gdsfactory without Anaconda3:

```
pip install gdsfactory
gf tool install
```

Or you can install the development version if you plan to [contribute](https://gdsfactory.readthedocs.io/en/latest/contribution.html) to gdsfactory:

```
git clone https://github.com/gdsfactory/gdsfactory.git
cd gdsfactory
make install
```

To summarize: There are 2 methods to install gdsfactory

1. `pip install gdsfactory` will download it from [PyPi (python package index)](https://pypi.org/project/gdsfactory/)
2. you can download it from [GitHub](https://pypi.org/project/gdsfactory/) in your computer and link the library to your python

```
git clone https://github.com/gdsfactory/gdsfactory.git
cd gdsfactory
make install
```

for updating 1. you need to `pip install gdsfactory --upgrade`
for updating 2. you need to pull from GitHub the latest changes

```
cd gdsfactory
git pull
```

After installing you should be able to `import gdsfactory as gf` from a python script.

- gdsfactory
  - components: define components
  - drc: check geometry
  - gdsdiff: hash geometry and show differences by displaying boolean operations in klayout
  - klive: stream GDS directly to klayout
  - ports: to connect components
  - routing: add waveguides to connect components
  - samples: python tutorial
  - tests:
  - klayout: klayout generic tech layers and klive macro
- docs/notebooks: jupyter-notebooks based tutorial

## Links

- [gdsfactory](https://github.com/gdsfactory/gdsfactory): Github repo where we store the gdsfactory code
- [gdslib](https://github.com/gdsfactory/gdslib): separate package for component circuit models (based on Sparameters).
- [ubc PDK](https://github.com/gdsfactory/ubc)
- [awesome photonics list](https://github.com/joamatab/awesome_photonics)
- [phidl (gdsfactory is based on phidl)](https://github.com/amccaugh/phidl)
- [gdspy (phidl is based on gdspy)](https://github.com/heitzmann/gdspy)
- [picwriter](https://github.com/DerekK88/PICwriter)
