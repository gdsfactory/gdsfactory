# gdsfactory 6.26.0

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gdsfactory.svg)](https://anaconda.org/conda-forge/gdsfactory)
[![Dockerhub](https://img.shields.io/docker/pulls/joamatab/gdsfactory)](https://hub.docker.com/r/joamatab/gdsfactory)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![issues](https://img.shields.io/github/issues/gdsfactory/gdsfactory)](https://github.com/gdsfactory/gdsfactory/issues)
[![forks](https://img.shields.io/github/forks/gdsfactory/gdsfactory.svg)](https://github.com/gdsfactory/gdsfactory/network/members)
[![GitHub stars](https://img.shields.io/github/stars/gdsfactory/gdsfactory.svg)](https://github.com/gdsfactory/gdsfactory/stargazers)
[![Downloads](https://pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![Downloads](https://pepy.tech/badge/gdsfactory/month)](https://pepy.tech/project/gdsfactory)
[![Downloads](https://pepy.tech/badge/gdsfactory/week)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gdsfactory/gdsfactory)

![logo](https://i.imgur.com/v4wpHpg.png)

GDSfactory is a design automation tool for photonics and analog circuits.

You can describe your circuits in code (python or YAML), verify them (DRC, simulation) and analyze them.

It provides you with an end to end flow for building chips.

![workflow](https://i.imgur.com/abvxJJw.png)


You can:

- Design (Layout, Simulation, Optimization)
    * define parametric cells (PCells) functions in python or YAML. Define routes between component ports.
    * Test component settings, ports and geometry to avoid unwanted regressions.
    * Capture design intent in a schematic.
- Verificate (DRC, DFM, LVS)
    * Run simulations directly from the layout thanks to the simulation interfaces. No need to draw the geometry more than once.
        - Run Component simulations (solve modes, FDTD, EME, TCAD, thermal ...)
        - Run Circuit simulations from the Component netlist (Sparameters, Spice ...)
        - Build Component models and study Design For Manufacturing.
    * Create DRC rule decks in Klayout.
    * Make sure complex layouts match their design intent (Layout Versus Schematic).
- Validate
    * Make sure that as you define the layout you define the test sequence, so when the chips come back you already know how to test them.
    * Model extraction: extract the important parameters for each component.
    * Build a data pipeline from raw data, to structured data and dashboards for monitoring your chip performance.


As input, you write python or YAML code.

As output you write a GDSII or OASIS file that you can send to your foundry for fabrication.
It also exports component settings (for measurement and data analysis) and netlists (for circuit simulations).

![layout_to_components](https://i.imgur.com/JLsvpLv.png)

![flow](https://i.imgur.com/XbhWJDz.png)


It provides you a common syntax for design (KLayout, gdstk, Ansys Lumerical, tidy3d, MEEP, MPB, DEVSIM, SAX, ...), verification and validation.

![tool interfaces](https://i.imgur.com/9fNLRvJ.png)


Multiple Silicon Photonics foundries have gdsfactory PDKs available. Talk to your foundry to access their gdsfactory PDK.

You can also access:

- open source PDKs available on GitHub
    * [UBCPDK](https://gdsfactory.github.io/ubc/README.html)
    * [skywater130](https://gdsfactory.github.io/skywater130/README.html)
- instructions on [how to build your own PDK](https://gdsfactory.github.io/gdsfactory/notebooks/08_pdk.html)
- instructions on [how to import a PDK from a library of fixed GDS cells](https://gdsfactory.github.io/gdsfactory/notebooks/09_pdk_import.html)


## Installation

### Installation for new python users

If you don't have python installed on your system you can [download the gdsfactory installer](https://github.com/gdsfactory/gdsfactory/releases) that includes python3, miniconda and all gdsfactory plugins.

### Installation for new gdsfactory users

Open Mamba, Miniconda or Anaconda Prompt and then install using pip

![anaconda prompt](https://i.imgur.com/Fyal5sT.png)


```
pip install gdsfactory --upgrade
gf tool install
```



### Installation for developers

For developers you need to fork the GitHub repository, git clone it (download it), git add, git commit, git push your improvement. Then pull request your changes to the main branch from the GitHub website.
For that you can install gdsfactory locally on your computer in `-e` edit mode.

```
git clone https://github.com/gdsfactory/gdsfactory.git
cd gdsfactory
pip install -e . pre-commit
pre-commit install
gf tool install
```

Then you need to restart Klayout to make sure you activate the klayout gdsfactory integration.

## Getting started

- Run notebooks on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)
- [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=250169028)
- [See slides](https://docs.google.com/presentation/d/1_ZmUxbaHWo_lQP17dlT1FWX-XD8D9w7-FcuEih48d_0/edit#slide=id.g11711f50935_0_5)
- [Read docs](https://gdsfactory.github.io/gdsfactory/)
- [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/watch?v=KXq09GirynI&list=PLZ3ZVd41isDDnuCirqIhNa8vsaHmbmxqM)
- [![Join the chat at https://gitter.im/gdsfactory-dev/community](https://badges.gitter.im/gdsfactory-dev/community.svg)](https://gitter.im/gdsfactory-dev/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
- See announcements on [GitHub](https://github.com/gdsfactory/gdsfactory/discussions/547), [google-groups](https://groups.google.com/g/gdsfactory) or [LinkedIn](https://www.linkedin.com/company/gdsfactory)


## Testimonals

"I've used **gdsfactory** since 2017 for all my chip tapeouts. I love that it is fast, easy to use, and easy to extend. It's the only tool that allows us to have an end-to-end chip design flow (design, verification and validation)."

<div style="text-align: right; margin-right: 10%;">Joaquin Matres - <strong>Google</strong>

---

"I've relied on **gdsfactory** for several tapeouts over the years. It's the only tool I've found that gives me the flexibility and scalability I need for a variety of projects."

<div style="text-align: right; margin-right: 10%;">Alec Hammond - <strong>Meta Reality Labs Research</strong>

---

## Acks

Contributors (in chronological order):

- Joaquin Matres (Google): write some documentation pages, help porting from gdspy to gdstk.
- Damien Bonneau (PsiQuantum): cell decorator, Component routing functions, Klayout placer.
- Pete Shadbolt (PsiQuantum): Klayout auto-placer, Klayout GDS interface (klive).
- Troy Tamas (Rockley): get_route_from_steps, netlist driven flow (from_yaml).
- Floris Laporte (Rockley): netlist extraction and circuit simulation interface with SAX.
- Alec Hammond (Meta Reality Labs Research): Meep and MPB interface.
- Simon Bilodeau (Princeton): Meep FDTD write Sparameters, TCAD device simulator.
- Thomas Dorch (Freedom Photonics): Meep's material database access, MPB sidewall angles, and add_pin_path.
- Jan-David Fischbach (Black semiconductor): improvements in pack_doe.
- Igal Bayn (Google): documentation improvements and suggestions.
- Alex Sludds (MIT): tiling fixes.
- Momchil Minkov (Flexcompute): improve tidy3d plugin.
- Skandan Chandrasekar (BYU): simphony, SiPANN plugins, A-star router.
- Helge Gehring (Google): simulation plugins (FEM heat solver), improving code quality and new components (spiral paths).
- Tim Ansell (Google): documentation improvements.
- Ardavan Oskoii (Google): Meep plugin documentation improvements.
- Marc de Cea (MIT): ge_detector, grating_coupler_dual, mmi_90degree_hybrid, coherent transceiver, receiver.
- Bradley Snyder (PHIX): grating_coupler snap to grid fixes.
- Jonathan Cauchon (Ciena): measurement database.
- Raphaël Dubé-Demers (EXFO): measurement database.
- Bohan Zhang (Boston University): grating coupler improvements.
- Niko Savola (IQM): optimization, notebook and code improvements.

Open source heroes:

- Matthias Köfferlein: Klayout
- Lucas Heitzmann (University of Campinas, Brazil): for gdstk
- Adam McCaughan (NIST): phidl. Inspiration for geometry manipulation.
- Alex Tait (Queens University): lytest inspiration for gdsfactory testing GDS framework.
- Thomas Ferreira de Lima (NEC): `pip install klayout` python API.
- Juan Sanchez: DEVSIM for TCAD simulations.
