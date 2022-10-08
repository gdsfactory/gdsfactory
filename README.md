# gdsfactory 5.40.0

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gdsfactory.svg)](https://anaconda.org/conda-forge/gdsfactory)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![issues](https://img.shields.io/github/issues/gdsfactory/gdsfactory)](https://github.com/gdsfactory/gdsfactory/issues)
[![forks](https://img.shields.io/github/forks/gdsfactory/gdsfactory.svg)](https://github.com/gdsfactory/gdsfactory/network/members)
[![GitHub stars](https://img.shields.io/github/stars/gdsfactory/gdsfactory.svg)](https://github.com/gdsfactory/gdsfactory/stargazers)
[![Downloads](https://pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![Downloads](https://pepy.tech/badge/gdsfactory/month)](https://pepy.tech/project/gdsfactory)
[![Downloads](https://pepy.tech/badge/gdsfactory/week)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/master/gdsfactory)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)

![logo](https://i.imgur.com/v4wpHpg.png)

GDSfactory is a design automation tool for photonics and analog circuits.

You can describe your circuits with a code driven flow (python or YAML), verify them (DRC, simulation) and analyze them.

Multiple Silicon Photonics foundries have gdsfactory PDKs available. Talk to your foundry to access their gdsfactory PDK.

You can also access:

- open source PDKs available on GitHub
    * [UBCPDK](https://gdsfactory.github.io/ubc/README.html)
    * [skywater130](https://gdsfactory.github.io/skywater130/README.html)
- instructions on [how to build your own PDK](https://gdsfactory.github.io/gdsfactory/notebooks/08_pdk.html)
- instructions on [how to import a PDK from a library of fixed GDS cells](https://gdsfactory.github.io/gdsfactory/notebooks/09_pdk_import.html)



You can:

- define parametric cells (PCells) in python or YAML.
- define routes between components.
- Test component settings, ports and geometry to avoid regressions.


As input, you write python or YAML code.

As output you write a GDSII or OASIS file that can send to your foundry.
It also exports component settings (for measurement and data analysis) and netlists (for circuit simulations).

![layout_to_components](https://i.imgur.com/JLsvpLv.png)

![flow](https://i.imgur.com/XbhWJDz.png)


It provides you a common syntax for layout (klayout, gdspy), simulation (Lumerical, tidy3d, MEEP, MPB, DEVSIM, simphony, SAX, ...) and data analysis libraries.

![tool interfaces](https://i.imgur.com/bQslWHO.png)


## Installation

[Download the latest installer](https://github.com/gdsfactory/gdsfactory/releases)

## Getting started

- Run notebooks on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)
- [See slides](https://docs.google.com/presentation/d/1_ZmUxbaHWo_lQP17dlT1FWX-XD8D9w7-FcuEih48d_0/edit#slide=id.g11711f50935_0_5)
- [Read docs](https://gdsfactory.github.io/gdsfactory/)
- [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/watch?v=KXq09GirynI&list=PLZ3ZVd41isDDnuCirqIhNa8vsaHmbmxqM)
- See announcenments on [GitHub](https://github.com/gdsfactory/gdsfactory/discussions/547), [google-groups](https://groups.google.com/g/gdsfactory) or [LinkedIn](https://www.linkedin.com/company/gdsfactory)

## Acks

Contributors (in chronological order):

- Joaquin Matres (Google): maintainer.
- Damien Bonneau (PsiQuantum): cell decorator, Component routing functions, Klayout placer.
- Pete Shadbolt (PsiQuantum): Klayout auto-placer, Klayout GDS interface (klive).
- Troy Tamas (Rockley): get_route_from_steps, netlist driven flow (from_yaml).
- Floris Laporte (Rockley): netlist extraction and circuit simulation interface with SAX.
- Alec Hammond (Meta Reality Labs Research): Meep and MPB interface.
- Simon Bilodeau (Princeton): Meep FDTD write Sparameters, TCAD device simulator.
- Thomas Dorch (Freedom Photonics): for Meep's material database access, MPB sidewall angles, and add_pin_path.
- Jan-David Fischbach (Black semiconductor): for improvements in pack_doe.
- Igal Bayn (Google): for documentation improvements and suggestions.
- Alex Sludds (MIT): for tiling fixes.
- Momchil Minkov (Flexcompute): for tidy3d plugin.
- Skandan Chandrasekar (BYU): for simphony, SiPANN plugins, A-star router.
- Helge Gehring (Google): for simulation plugins (FEM heat solver), improving code quality and new components (spiral paths).
- Tim Ansell (Google): for documentation improvements.
- Ardavan Oskoii (Google): for Meep plugin documentation improvements.
- Marc de Cea (MIT): for ge_detector, grating_coupler_dual, mmi_90degree_hybrid, coherent transceiver, receiver.
- Bradley Snyder (PHIX): for grating_coupler snap to grid fixes.
- Jonathan Cauchon (EHVA): for measurement database.
- Raphaël Dubé-Demers (EHVA): for measurement database.

Open source heroes:

- Matthias Köfferlein: for Klayout
- Lucas Heitzmann (University of Campinas, Brazil): for gdspy
- Adam McCaughan (NIST): for phidl. Inpiration for geometry manipulation.
- Alex Tait (Queens University): for lytest
- Thomas Ferreira de Lima (NEC): for `pip install klayout` python API.
- Juan Sanchez: for DEVSIM
