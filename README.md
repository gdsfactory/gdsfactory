# gdsfactory 8.4.0

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gdsfactory/gdsfactory-photonics-training)

![logo](https://i.imgur.com/cN1ZWq8.png)


> 🚀 **Notice: Major Release**
>
> We are excited to announce that gdsfactory has upgraded its backend from gdstk to KLayout. This change brings enhanced routing functions and additional features from KLayout, including DRC, dummy fill, and connectivity checks.
>
> Notice that the routing and some advanced functions have changed. For a complete list of changes, please refer to our [migration guide](https://gdsfactory.github.io/gdsfactory/notebooks/21_migration_guide_7_8.html) or review the updated layout tutorial.


Gdsfactory is a Python library for designing chips (Photonics, Analog, Quantum, MEMs, and more), 3D printed objects, and PCBs.
Here, you can code your designs in Python or YAML, perform verification (DRC, simulation, and extraction), and enable automated testing in the lab to ensure your fabricated devices meet your specifications.


As input you write python code, as an output GDSFactory creates CAD files (GDS, OASIS, STL, GERBER).

![cad](https://i.imgur.com/TWP80ez.png)

Highlights:

- More than 1M downloads
- More than 60 Contributors
- More than 10 PDKs available

![workflow](https://i.imgur.com/abvxJJw.png)

We facilitate an end-to-end design flow for you to:

- **Design (Layout, Simulation, Optimization)**: Define parametric cell functions in Python or YAML to generate components. Test component settings, ports, and geometry to avoid unwanted regressions, and capture design intent in a schematic.
- **Verify (DRC, DFM, LVS)**: Run simulations directly from the layout using our simulation interfaces, removing the need to duplicate geometry drawings. Conduct component and circuit simulations, study design for manufacturing, and ensure complex layouts match their design intent through Layout Versus Schematic verification.
- **Validate**: Define layout and test protocols simultaneously for automated chip analysis post-fabrication. This allows you to extract essential component parameters, and build data pipelines from raw data to structured data to monitor chip performance.

Your input: Python or YAML text.
Your output: A GDSII or OASIS file for fabrication, alongside component settings (for measurement and data analysis) and netlists (for circuit simulations) in YAML.

![layout_to_components](https://i.imgur.com/S96RSil.png)

![flow](https://i.imgur.com/XbhWJDz.png)

We provide a common syntax for design (KLayout, Ansys Lumerical, tidy3d, MEEP, MPB, DEVSIM, SAX, MEOW ...), verification, and validation.

![tool interfaces](https://i.imgur.com/ef26jbe.png)

Many foundries have gdsfactory PDKs available. Please to contact your foundry to access their gdsfactory PDK, as you will require an NDA:

- AIM photonics PDK
- AMF photonics PDK
- Compoundtek photonics PDK
- GlobalFoundries 45SPCLO Photonics PDK
- HHI Photonics PDK
- IMEC photonics PDK
- Ligentec PDK
- Lionix PDK
- Smart Photonics Photonics PDK
- TowerSemi PH18 photonics PDK
- III-V Labs PDK

There are some open source PDKs available without an NDA:

- [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/) (open source)
- [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc) (open source)
- [Skywater130 CMOS PDK](https://gdsfactory.github.io/skywater130) (open source)
- [VTT](https://github.com/gdsfactory/vtt) (open source)
- [Cornerstone](https://github.com/gdsfactory/cspdk) (open source)

You can also access:

- instructions on [how to build your own PDK](https://gdsfactory.github.io/gdsfactory/notebooks/08_pdk.html)
- instructions on [how to import a PDK from a library of fixed GDS cells](https://gdsfactory.github.io/gdsfactory/notebooks/09_pdk_import.html)

![pdks](https://i.imgur.com/7O011v0.png)

## Getting started

- [See slides](https://docs.google.com/presentation/d/1_ZmUxbaHWo_lQP17dlT1FWX-XD8D9w7-FcuEih48d_0/edit#slide=id.g11711f50935_0_5)
- [Read docs](https://gdsfactory.github.io/gdsfactory/)
- [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory625/playlists)
- [![Join the chat at https://gitter.im/gdsfactory-dev/community](https://badges.gitter.im/gdsfactory-dev/community.svg)](https://gitter.im/gdsfactory-dev/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
- See announcements on [GitHub](https://github.com/gdsfactory/gdsfactory/discussions/547), [google-groups](https://groups.google.com/g/gdsfactory) or [LinkedIn](https://www.linkedin.com/company/gdsfactory)
- [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gdsfactory/gdsfactory-photonics-training)
- [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=250169028)
- [PIC training](https://gdsfactory.github.io/gdsfactory-photonics-training/)
- Online course [UBCx: Silicon Photonics Design, Fabrication and Data Analysis](https://www.edx.org/learn/engineering/university-of-british-columbia-silicon-photonics-design-fabrication-and-data-ana), where students can use gdsfactory to create a design, have it fabricated, and tested.

## Who is using gdsfactory?

Hundreds of organisations are using gdsfactory. Some companies and organizations around the world using gdsfactory include:

![logos](https://i.imgur.com/IqTUq9S.png)

"I've used **gdsfactory** since 2017 for all my chip tapeouts. I love that it is fast, easy to use, and easy to extend. It's the only tool that allows us to have an end-to-end chip design flow (design, verification and validation)."

<div style="text-align: right; margin-right: 10%;">Joaquin Matres - <strong>Google</strong></div>

---

"I've relied on **gdsfactory** for several tapeouts over the years. It's the only tool I've found that gives me the flexibility and scalability I need for a variety of projects."

<div style="text-align: right; margin-right: 10%;">Alec Hammond - <strong>Meta Reality Labs Research</strong></div>

---

"The best photonics layout tool I've used so far and it is leaps and bounds ahead of any commercial alternatives out there. Feels like gdsfactory is freeing photonics."

<div style="text-align: right; margin-right: 10%;">Hasitha Jayatilleka - <strong>LightIC Technologies</strong></div>

---

"As an academic working on large scale silicon photonics at CMOS foundries I've used gdsfactory to go from nothing to full-reticle layouts rapidly (in a few days). I particularly appreciate the full-system approach to photonics, with my layout being connected to circuit simulators which are then connected to device simulators. Moving from legacy tools such as gdspy and phidl to gdsfactory has sped up my workflow at least an order of magnitude."

<div style="text-align: right; margin-right: 10%;">Alex Sludds - <strong>MIT</strong></div>

---

"I use gdsfactory for all of my photonic tape-outs. The Python interface makes it easy to version control individual photonic components as well as entire layouts, while integrating seamlessly with KLayout and most standard photonic simulation tools, both open-source and commercial.

<div style="text-align: right; margin-right: 10%;">Thomas Dorch - <strong>Freedom Photonics</strong></div>

## Why use gdsfactory?

- It's fast, extensible and easy to use.
- It's free, as in freedom and in cost.
- It's the most popular EDA tool with a growing community of users, developers, and extensions to other tools.

Gdsfactory is really fast thanks to KLayout C++ library for manipulating GDSII objects. You will notice this when reading/writing big GDS files or doing large boolean operations.

| Benchmark      |  gdspy  | gdsfactory | Gain |
| :------------- | :-----: | :--------: | :--: |
| 10k_rectangles | 80.2 ms |  4.87 ms   | 16.5 |
| boolean-offset | 187 μs  |  44.7 μs   | 4.19 |
| bounding_box   | 36.7 ms |   170 μs   | 216  |
| flatten        | 465 μs  |  8.17 μs   | 56.9 |
| read_gds       | 2.68 ms |   94 μs    | 28.5 |

## Contributors

Thanks to all the contributors that make this awesome project possible!

[![Meet our contributors!](https://contrib.rocks/image?repo=gdsfactory/gdsfactory)](https://github.com/gdsfactory/gdsfactory/graphs/contributors)
