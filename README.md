# gdsfactory 3.10.7

![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)
[![](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![](https://img.shields.io/github/issues/gdsfactory/gdsfactory)](https://github.com/gdsfactory/gdsfactory/issues)
![](https://img.shields.io/github/forks/gdsfactory/gdsfactory)
![](https://img.shields.io/github/stars/gdsfactory/gdsfactory)
[![](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/master/gdsfactory)
[![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![](https://i.imgur.com/v4wpHpg.png)

gdsfactory is an [EDA (electronics design automation)](https://en.wikipedia.org/wiki/Electronic_design_automation) tool to Layout Integrated Circuits.

It is build on top of [phidl](https://github.com/amccaugh/phidl), [gdspy](https://github.com/heitzmann/gdspy) and klayout to provide you with functions to build your GDSII components, PDKs and masks for different foundries.

You just need to adapt the functions to your foundry and build your own library of elements (see [UBC PDK](https://github.com/gdsfactory/ubc) example).

gdsfactory provides you with functions that you can use to:

- define components, circuits and masks in python or YAML
- add routes between components
- ensure you can build complex systems by testing settings, ports and GDS geometry

As input, gdsfactory needs you to write python or YAML code to describe your layouts.

As output it creates a [GDSII file](https://en.wikipedia.org/wiki/GDSII) which is the most common filetype used by CMOS foundries.
It also can output components settings (that you can use for measurement and data analysis) or netlists (for circuit simulations). You can also easily adapt this metadata output files to your needs.

![](https://i.imgur.com/XbhWJDz.png)

## Documentation

For Photonics IC layout I used [IPKISS](https://github.com/jtambasco/ipkiss) for 7 years.

IPKISS is quite slow when working with large layouts, so in 2019 I stopped using it.

Then I tried all the commercial (Luceda, Cadence, Synopsys) and open source EDA tools (phidl, gdspy, picwriter, klayout-zero-pdk, nazca) looking for a fast and easy to use workflow.

The metrics for the benchmark were:

1. Fast
2. Easy to use
3. Maintained / Documented / Popular


PHIDL was the outstanding winner thanks to its speed, code beauty and easy of use, and is written on top of gdspy (which came second), so you can also leverage all the goodies from gdspy.

Gdsfactory also leverages klayout and gdspy libraries for some functions.

![](https://i.imgur.com/4xQJ2yk.png)

What nice things come from phidl?

- functional programming that follow UNIX philosophy
- nice API to create and modify Components
- Easy definition of paths, cross-sections and extrude them into Components
- Easy definition of ports, to connect components. Ports in phidl have name, position, width and orientation (in degrees)
  - gdsfactory expands phidl ports with layer, port_type (optical, electrical, vertical_te, vertical_tm ...) and cross_section
  - gdsfactory adds renaming ports functions (clockwise, counter_clockwise ...)

What nice things come from klayout?

- GDS viewer. gdsfactory can send GDS files directly to klayout, you just need to have klayout open
- layer colormaps for showing in klayout, matplotlib, trimesh (using the same colors)
- fast boolean xor to avoid geometric regressions on Components geometry. Klayout booleans are faster than gdspy ones
- basic DRC checks

What functionality does gdsfactory provide you on top phidl/gdspy/klayout?

- `@cell decorator` for decorating functions that create components
  - autonames Components with a unique name that depends on the input parameters
  - avoids duplicated names and faster runtime implementing a cache. If you try to call the same component function with the same parameters, you get the component directly from the cache.
  - automatically adds cell parameters into a `component.info` (`full`, `default`, `changed`) as well as any other `info` metadata (`polarization`, `wavelength`, `test_protocol`, `simulation_settings` ...)
  - writes component metadata in YAML including port information (name, position, width, orientation, type, layer)
- routing functions where the routes are composed of configurable bends and straight sections (for circuit simulations you want to maintain the route bends and straight settings)
  - `get_route`: for single routes between component ports
  - `get_route_from_steps`: for single routes between ports where we define the steps or bends
  - `get_bundle`: for bundles of routes (river routing)
  - `get_bundle_path_length_match`: for routes that need to keep the same path length
  - `get_route(auto_widen=True)`: for routes that expand to wider waveguides to reduce loss and phase errors
  - `get_route(impossible route)`: for impossible routes it raises a warning and returns a FlexPath on an error layer
- testing framework to avoid unwanted regressions
  - checks geometric GDS changes by making a boolean difference between GDS cells
  - checks metadata changes, including port location and component settings
- large library of photonics and electrical components that you can easily customize to your technology
- read components from GDS, numpy, YAML
- export components to GDS, YAML or 3D (trimesh, STL ...)
- export netlist in YAML format
- plugins to compute Sparameters using for example Lumerical, meep or tidy3d

How can you learn more?

gdsfactory is written in python and requires some basic knowledge of python. If you are new to python you can find many [books](https://jakevdp.github.io/PythonDataScienceHandbook/index.html), [youTube videos](https://www.youtube.com/c/anthonywritescode) and [courses](https://github.com/joamatab/practical-python) available online.

Once you are familiar with python, you can also:

- [read online docs](https://gdsfactory.github.io/gdsfactory/)
- run gdsfactory/samples
- run [docs/notebooks](https://gdsfactory.readthedocs.io/en/latest/notebooks.html)

## Installation

First, you need to install [klayout](https://www.klayout.de/) to visualize the GDS files that you create.

gdsfactory works for python>=3.7 in Windows, MacOs and Linux.
[Github](https://github.com/gdsfactory/gdsfactory/actions) runs all the tests at least once a day for different versions of python (3.7, 3.8, 3.9)

If you are on Windows, I recommend you install gdsfactory with Anaconda3 or Miniconda3.

```
conda install -c conda-forge gdspy
conda install -c conda-forge triangle
pip install gdsfactory[full]
gf tool install
```

For Linux and MacOs you can also install gdsfactory without Anaconda3:

```
pip install gdsfactory[pip]
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
  - components: define a basic library of generic components that you can customize
  - gdsdiff: hash geometry and show differences by displaying boolean operations in klayout
  - klayout: klayout generic tech layers and klive macro
  - klive: stream GDS directly to klayout
  - ports: to connect components
  - routing: add waveguides to connect components
  - samples: python tutorial
  - tests:
- docs/notebooks: jupyter-notebooks based tutorial

## Plugins

We try to keep gdsfactory core with minimum dependencies.
So when you run `pip install gdsfactory` you do not install any plugins by default.
If you want to install gdsfactory together with all the plugins you can run

```
pip install gdsfactory[full]
```

### Trimesh

For (3D rendering and STL export)

### meep / mpb

Open source FDTD / mode simulator. Requires you to run `conda install -c conda-forge pymeep` (or `conda install -c conda-forge pymeep=*=mpi_mpich_*` for parallel execution).

Also for parallel execution you can install `conda install mpi4py -y`

### tidy3d

For FDTD simulations on the web. Requires you to create an account on [simulation.cloud](simulation.cloud)

## CHANGELOG

gdsfactory keeps a general [changelog](https://gdsfactory.readthedocs.io/en/latest/CHANGELOG.html) as well as for each individual component

By default each component has an empty changelog and starts with version '0.0.1'

- major:
  - position of ports changes. Affects position of ports. High level circuits may not work.
- minor
  - affects a physical layer. Footprint is the same. it fits, but may not work the same.
- patch: (decoration or formalities)
  - smaller changes
  - name change

You will need to manually bump the version and document any changes

```
@gf.cell
def my_component():
    c = gf.Component()

    c.version = "0.0.2"
    c.changelog += " # 0.0.2 increase default length to 11um "
```

## Acks

gdsfactory top contributors:

- Joaquin Matres (Google): maintainer
- Damien Bonneau (PsiQ): cell decorator, Component routing functions, Klayout placer
- Pete Shadbolt (PsiQ): Klayout auto-placer, Klayout GDS interface (klive)
- Troy Tamas (Rockley): get_route_from_steps, component_from_yaml (schematic capture)
- Floris Laporte (Rockley): netlist extraction and circuit simulation interface with [SAX](https://sax.readthedocs.io/en/latest/)
- Alec Hammond (Georgia Tech): Meep and MPB interface
- Simon Biladieu (Princeton): Meep FDTD write Sparameters
- Thomas Dorch (Freedom Photonics): for Meep's material database access and MPB sidewall angles

Open source heroes:

- Matthias Köfferlein (Germany): for Klayout
- Lucas Heitzmann (University of Campinas, Brazil): for gdspy
- Adam McCaughan (NIST): for phidl
- Alex Tait (Queens University): for lytest
- Thomas Ferreira de Lima (NEC): for `pip install klayout`


## Links

- [gdsfactory github repo](https://github.com/gdsfactory/gdsfactory) and [docs](https://gdsfactory.github.io/gdsfactory/)
- [gdslib](https://github.com/gdsfactory/gdslib): separate package for component circuit models (based on Sparameters).
- [ubc PDK](https://github.com/gdsfactory/ubc): sample open source PDK from edx course.
- [awesome photonics list](https://github.com/joamatab/awesome_photonics)
- [phidl (gdsfactory is based on phidl)](https://github.com/amccaugh/phidl)
- [gdspy (phidl is based on gdspy)](https://github.com/heitzmann/gdspy)
- [picwriter](https://github.com/DerekK88/PICwriter)
- [docs follow MyST syntax](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html)
- [versions follow semantic versioning](https://semver.org/)
