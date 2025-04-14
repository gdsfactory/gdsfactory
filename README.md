# GDSFactory 9.4.0

[![docs](https://github.com/gdsfactory/gdsfactory/actions/workflows/pages.yml/badge.svg)](https://gdsfactory.github.io/gdsfactory/)
[![PyPI](https://img.shields.io/pypi/v/gdsfactory)](https://pypi.org/project/gdsfactory/)
[![PyPI Python](https://img.shields.io/pypi/pyversions/gdsfactory.svg)](https://pypi.python.org/pypi/gdsfactory)
[![Downloads](https://static.pepy.tech/badge/gdsfactory)](https://pepy.tech/project/gdsfactory)
[![MIT](https://img.shields.io/github/license/gdsfactory/gdsfactory)](https://choosealicense.com/licenses/mit/)
[![codecov](https://img.shields.io/codecov/c/github/gdsfactory/gdsfactory)](https://codecov.io/gh/gdsfactory/gdsfactory/tree/main/gdsfactory)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gdsfactory/binder-sandbox/HEAD)


As input you write python code, as an output GDSFactory creates CAD files (GDS, OASIS, STL, GERBER).

![cad](https://i.imgur.com/3cUa2GV.png)

```python
import gdsfactory as gf

c = gf.Component()
ref1 = c.add_ref(gf.components.rectangle(size=(10, 10), layer=(1, 0)))
ref2 = c.add_ref(gf.components.text("Hello", size=10, layer=(2, 0)))
ref3 = c.add_ref(gf.components.text("world", size=10, layer=(2, 0)))

ref1.xmax = ref2.xmin - 5
ref3.xmin = ref2.xmax + 2
ref3.rotate(30)
c.show()
```

Highlights:

- +2M downloads
- +79 Contributors
- +25 PDKs available

![workflow](https://i.imgur.com/KyavbHh.png)

We provide a comprehensive end-to-end design flow that enables you to:

- **Design (Layout, Simulation, Optimization)**: Define parametric cell functions in Python to generate components. Test component settings, ports, and geometry to avoid unwanted regressions, and capture design intent in a schematic.
- **Verify (DRC, DFM, LVS)**: Run simulations directly from the layout using our simulation interfaces, removing the need to redraw your components in simulation tools. Conduct component and circuit simulations, study design for manufacturing. Ensure complex layouts match their design intent through Layout Versus Schematic verification (LVS) and are DRC clean.
- **Validate**: Define layout and test protocols simultaneously for automated chip analysis post-fabrication. This allows you to extract essential component parameters, and build data pipelines from raw data to structured data to monitor chip performance.

Your input: Python or YAML text.
Your output: A GDSII or OASIS file for fabrication, alongside component settings (for measurement and data analysis) and netlists (for circuit simulations) in YAML.

We provide a common syntax for design (Ansys, Lumerical, Tidy3d, MEEP, DEVSIM, SAX, MEOW, Xyce ...), verification, and validation.

![tool interfaces](https://i.imgur.com/j5qlFWj.png)

## GDSFactory+

**GDSFactory+** offers Graphical User Interface for chip design, built on top of GDSFactory and VSCode. It provides you:

- Foundry PDK access
- Schematic capture
- Simulations
- Design verification (DRC, LVS)
- Data analytics

## Accessing Foundry PDKs

Access to GDSFactory PDKs under NDA requires a GDSFactory+ subscription.

To sign up, visit [GDSFactory.com](https://gdsfactory.com/). Once registered, you can request access to foundry PDKs that require an NDA with the respective foundry.

Available Foundry PDKs under NDA:

- AIM photonics PDK
- AMF photonics PDK
- Compoundtek photonics PDK
- Fraunhofer HHI Photonics PDK
- Smart Photonics PDK
- TowerSemi PH18 photonics PDK
- TowerSemi PH18DA photonics PDK by OpenLight
- III-V Labs PDK
- Lionix PDK
- Ligentec PDK
- Lightium PDK
- QCI (Quantum Computing Inc)

There are also **open-source PDKs** available that do not require an NDA:

- [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/)
- [ANT / SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc)
- [Skywater130 CMOS PDK](https://gdsfactory.github.io/skywater130)
- [VTT](https://github.com/gdsfactory/vtt)
- [Cornerstone](https://github.com/gdsfactory/cspdk)
- [Luxtelligence](https://github.com/Luxtelligence/lxt_pdk_gf)

## Getting started

- [See slides](https://docs.google.com/presentation/d/1_ZmUxbaHWo_lQP17dlT1FWX-XD8D9w7-FcuEih48d_0/edit#slide=id.g11711f50935_0_5)
- [Read docs](https://gdsfactory.github.io/gdsfactory/)
- [![Video Tutorials](https://img.shields.io/badge/youtube-Video_Tutorials-red.svg?logo=youtube)](https://www.youtube.com/@gdsfactory/playlists)
- [![Join the chat at https://gitter.im/gdsfactory-dev/community](https://badges.gitter.im/gdsfactory-dev/community.svg)](https://gitter.im/gdsfactory-dev/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
- See announcements on [GitHub](https://github.com/gdsfactory/gdsfactory/discussions/547), [google-groups](https://groups.google.com/g/gdsfactory) or [LinkedIn](https://www.linkedin.com/company/gdsfactory)
- [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=250169028)
- [PIC training](https://gdsfactory.github.io/gdsfactory-photonics-training/)
- Online course [UBCx: Silicon Photonics Design, Fabrication and Data Analysis](https://www.edx.org/learn/engineering/university-of-british-columbia-silicon-photonics-design-fabrication-and-data-ana), where students can use GDSFactory to create a design, have it fabricated, and tested.
- [Visit website](https://gdsfactory.com)

## Who is using GDSFactory?

Hundreds of organisations are using GDSFactory. Some companies and organizations around the world using GDSFactory include:

![logos](https://i.imgur.com/VzLNMH1.png)

"I've used **GDSFactory** since 2017 for all my chip tapeouts. I love that it is fast, easy to use, and easy to extend. It's the only tool that allows us to have an end-to-end chip design flow (design, verification and validation)."

<div style="text-align: right; margin-right: 10%;">Joaquin Matres - <strong>Google</strong></div>

---

"I've relied on **GDSFactory** for several tapeouts over the years. It's the only tool I've found that gives me the flexibility and scalability I need for a variety of projects."

<div style="text-align: right; margin-right: 10%;">Alec Hammond - <strong>Meta Reality Labs Research</strong></div>

---

"The best photonics layout tool I've used so far and it is leaps and bounds ahead of any commercial alternatives out there. Feels like GDSFactory is freeing photonics."

<div style="text-align: right; margin-right: 10%;">Hasitha Jayatilleka - <strong>LightIC Technologies</strong></div>

---

"As an academic working on large scale silicon photonics at CMOS foundries I've used GDSFactory to go from nothing to full-reticle layouts rapidly (in a few days). I particularly appreciate the full-system approach to photonics, with my layout being connected to circuit simulators which are then connected to device simulators. Moving from legacy tools such as gdspy and phidl to GDSFactory has sped up my workflow at least an order of magnitude."

<div style="text-align: right; margin-right: 10%;">Alex Sludds - <strong>MIT</strong></div>

---

"I use GDSFactory for all of my photonic tape-outs. The Python interface makes it easy to version control individual photonic components as well as entire layouts, while integrating seamlessly with KLayout and most standard photonic simulation tools, both open-source and commercial.

<div style="text-align: right; margin-right: 10%;">Thomas Dorch - <strong>Freedom Photonics</strong></div>

## Why Use GDSFactory?

- **Fast, extensible, and easy to use** – designed for efficiency and flexibility.
- **Free and open-source** – no licensing fees, giving you the freedom to modify and extend it.
- **A thriving ecosystem** – the most popular EDA tool with a growing community of users, developers, and integrations with other tools.
- **Built on the open-source advantage** – just like the best machine learning libraries, GDSFactory benefits from continuous contributions, transparency, and innovation.

GDSFactory is really fast thanks to KLayout C++ library for manipulating GDS objects. You will notice this when reading/writing big GDS files or doing large boolean operations.

| Benchmark      |  gdspy  | GDSFactory | Gain |
| :------------- | :-----: | :--------: | :--: |
| 10k_rectangles | 80.2 ms |  4.87 ms   | 16.5 |
| boolean-offset | 187 μs  |  44.7 μs   | 4.19 |
| bounding_box   | 36.7 ms |   170 μs   | 216  |
| flatten        | 465 μs  |  8.17 μs   | 56.9 |
| read_gds       | 2.68 ms |   94 μs    | 28.5 |

## Contributors

A huge thanks to all the contributors who make this project possible!

We welcome all contributions—whether you're adding new features, improving documentation, or even fixing a small typo. Every contribution helps make GDSFactory better!

Join us and be part of the community. 🚀

![contributors](https://i.imgur.com/0AuMHZE.png)


## Stargazers

[![Stargazers over time](https://starchart.cc/gdsfactory/gdsfactory.svg)](https://starchart.cc/gdsfactory/gdsfactory)
