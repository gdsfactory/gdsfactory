# gdsfactory 5.9.0

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![pypi](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![issues](https://img.shields.io/github/issues/gdsfactory/gdsfactory)](https://github.com/gdsfactory/gdsfactory/issues)
[![forks](https://img.shields.io/github/forks/gdsfactory/gdsfactory.svg)](https://github.com/gdsfactory/gdsfactory/network/members)
[![GitHub stars](https://img.shields.io/github/stars/gdsfactory/gdsfactory.svg)](https://github.com/gdsfactory/gdsfactory/stargazers)
[![Downloads](https://pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![Downloads](https://pepy.tech/badge/gdsfactory/month)](https://pepy.tech/project/gdsfactory)
[![Downloads](https://pepy.tech/badge/gdsfactory/week)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/master/gdsfactory)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/gdsfactory/HEAD)

![](https://i.imgur.com/v4wpHpg.png)

gdsfactory is an EDA (electronics design automation) tool to Layout Integrated Circuits.
It is built on top of phidl, gdspy and klayout to work with GDSII components, PDKs and masks for different foundries.
It combines the power of a code driven flow (python or YAML) together with visualization (Klayout for GDS, trimesh for 3D rendering, networkx for graphs ...) and simulation (for component and circuit) interfaces.

You just need to adapt the functions to your foundry and build your own library of elements (see [UBC PDK](https://github.com/gdsfactory/ubc) example).

gdsfactory provides you with functions that you can use to:

- define components, circuits and masks in python or YAML
- route between components
- test settings, ports and GDS geometry

It enables both layout and netlist driven flows and is all code driven.

As input, you write python or YAML code.

As output it creates a GDSII file which is the most common file format used by CMOS foundries.
It also can output components settings (that you can use for measurement and data analysis) or netlists (for circuit simulations). And you can easily adapt any outputs to your needs, thanks to being all natively written in python.

![](https://i.imgur.com/XbhWJDz.png)


gdsfactory is based on phidl, gdspy and klayout.

![](https://i.imgur.com/zMpvrWr.png)


## Getting started

- Run notebooks on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/gdsfactory/HEAD)
- [see slides](https://docs.google.com/presentation/d/1_ZmUxbaHWo_lQP17dlT1FWX-XD8D9w7-FcuEih48d_0/edit#slide=id.g11711f50935_0_5)
- [read docs](https://gdsfactory.github.io/gdsfactory/)

## Acks

gdsfactory top contributors:

- Joaquin Matres (Google): maintainer
- Damien Bonneau (PsiQ): cell decorator, Component routing functions, Klayout placer
- Pete Shadbolt (PsiQ): Klayout auto-placer, Klayout GDS interface (klive)
- Troy Tamas (Rockley): get_route_from_steps, netlist driven flow (from_yaml)
- Floris Laporte (Rockley): netlist extraction and circuit simulation interface with [SAX](https://flaport.github.io/sax)
- Alec Hammond (Georgia Tech): Meep and MPB interface
- Simon Bilodeau (Princeton): Meep FDTD write Sparameters
- Thomas Dorch (Freedom Photonics): for Meep's material database access, MPB sidewall angles, and add_pin_path
- Igal Bayn (Google): for documentation improvements and suggestions.
- Alex Sludds (MIT): for tiling fixes.

Open source heroes:

- Matthias Köfferlein (Germany): for Klayout
- Lucas Heitzmann (University of Campinas, Brazil): for gdspy
- Adam McCaughan (NIST): for phidl
- Alex Tait (Queens University): for lytest
- Thomas Ferreira de Lima (NEC): for `pip install klayout`

## Links

- [gdsfactory github repo](https://github.com/gdsfactory/gdsfactory) and [docs](https://gdsfactory.github.io/gdsfactory/)
- [ubc PDK](https://github.com/gdsfactory/ubc): sample open source PDK from edx course.
- [awesome photonics list](https://github.com/joamatab/awesome_photonics)
