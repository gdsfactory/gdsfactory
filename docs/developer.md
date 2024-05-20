# Installation

We support Python 3.10 or 3.11, and recommend [VSCode](https://code.visualstudio.com/) IDE. If you do not have Python installed, you can [download Anaconda](https://www.anaconda.com/download/).

Upon Python installation, open Anaconda Prompt as Administrator and install the latest gdsfactory

![anaconda prompt](https://i.imgur.com/eKk2bbs.png)


```
pip install gdsfactory --upgrade
```

## Update gdsfactory

You can upgrade gdsfactory using the following command:

```
pip install gdsfactory --upgrade
```

Please note that some PDKs may only work for a specific version of gdsfactory. Ensure you install the correct gdsfactory version specified in the pyproject.toml file. This will automatically happen when you install gdsfactory as one of the PDK dependencies. For example, pip install gf180 will install the latest gdsfactory version tested for the GlobalFoundries180 PDK.

To determine your python and gdsfactory versions, use the following code:

```
import sys
print(sys.version)
print(sys.executable)

import gdsfactory as gf
gf.config.print_version_plugins()
```

## Docker container

As an alternative, you can use the pre-built Docker image from [github](https://github.com/gdsfactory/gdsfactory/pkgs/container/gdsfactory) or [hub.docker.com/r/joamatab/gdsfactory](https://hub.docker.com/r/joamatab/gdsfactory)

For instance, VS Code supports development inside a container. See [Developing inside a Container](https://code.visualstudio.com/docs/devcontainers/containers) for details.

## Installation for developers

As a contributor, if you are on windows you need to download [Git](https://git-scm.com/download/win) and optionally [GitHub Desktop](https://desktop.github.com/).

Then you need to fork the [GitHub repository](https://github.com/gdsfactory/gdsfactory), git clone it (download it), git add, git commit, git push your improvement. Then pull request your changes to the main branch from the GitHub website.

The following lines will:

- clone your gdsfactory fork (make sure you change `YourUserName` with your GitHub user name)
- download the GDS golden data for running GDS regressions from a separate [repo](https://github.com/gdsfactory/gdsfactory-test-data/tree/test-data)
- install gdsfactory locally on your computer in `-e` edit mode.
- install pre-commit hooks for making sure your code syntax and style matches some basic rules.

```
git clone git@github.com:YourUserName/gdsfactory.git
cd gdsfactory
git clone https://github.com/gdsfactory/gdsfactory-test-data.git -b test-data test-data
pip install -e .[dev]
pre-commit install
```

**Note**:
- If you need to run the notebooks and you are not using VSCode or Anaconda, then you need to install [jupyter lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
```
pip install jupyterlab
```

## Style

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and existing code.
- Make sure tests pass on GitHub.
- Install pre-commit to get the pre-commit checks passing (autoformat the code, run linter ...).

Pre-commit makes sure your code is formatted following black and checks syntax.
If you forgot to `pre-commit install` you can fix pre-commit issues by running

```
pre-commit run --all-files
```

until you fix all the issues that the pre-commit check complaints about.

## Tests

gdsfactory tests are written with [pytest](https://docs.pytest.org/en/latest/contents.html). You can run them, from the root of the repository with pytest:

```shell
pytest -s
```

pytest will test any function that starts with `test_`. You can assert the number of polygons, the name, the length of a route or whatever you want.

In addition to unit tests run against the library, gdsfactory has a suite of regression tests which ensure that Components are never unintentionally modified between revisions. These regression tests include
| Test Type | Path | Format | Purpose |
|------|------|---------|--|
| GDS | `tests/components/test_components.py:test_gds` | GDS | Tests that GDS files have not changed either structurally (cell names and hierarchy) or geometrically (XOR). |
| Settings | `tests/components/test_components.py:test_settings` | YAML | Tests that component settings have not changed. |
| Netlist | `tests/test_netlists.py` | YAML | Tests that extracted netlist yaml contents have not changed. |
| Ports | `tests/components/test_ports.py` | CSV | Tests that port locations have not changed |
| Containers | `tests/test_containers.py` | YAML | Tests that container settings have not changed |

- regressions tests: avoids unwanted regressions by storing Components port locations in CSV and metadata in YAML files. You can force to regenerate the reference files running `pytest --force-regen -s` from the repo root directory.
  - `tests/components/test_components.py` stores all the component settings in YAML
  - `tests/components/test_ports.py` stores all port locations in a CSV file
  - `tests/test_netlists.py` stores all the component netlist in YAML and rebuilds the component from the netlist. Converts the routed PIC into YAML and build back into the same PIC from its YAML definition
  - difftest: writes all components GDS in `run_layouts` and compares them with `ref_layouts`. When running the test it will do a boolean of the `run_layout` and the `ref_layout` and raise an error for any significant differences. It will prompt you to review the differences in klayout and approve or reject the new GDS.

To regenerate regression reference files, you can run

```shell
pytest --force-regen -s
```

Note that the `--force-regen` flag will regenerate textual reference files, via [pytest-regressions](https://pytest-regressions.readthedocs.io/en/latest/overview.html). When GDS file regressions are found, the `-s` flag will cause pytest to step through the failures one-by-one, so you can inspect the XOR result in Klayout (automatically loaded via klive) and debug messages in the terminal. You will be prompted if you would like to accept or reject the set of changes for each file.

## Build your own Reticles/projects/PDKs

We recommend creating a separate python project for each mask and PDK.

```
pip install cookiecutter
cookiecutter gh:joamatab/python
```

Make sure you pin the version of gdsfactory that you are using in your `pyproject.toml`

## Test own Reticles/projects/PDKs

You can take a look at the tests of other open source PDKs.

- [SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc) (open source)
- [VTT photonics PDK](https://gdsfactory.github.io/vtt) (open source)
- [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/) (open source)
- [Skywater130 CMOS PDK](https://gdsfactory.github.io/skywater130) (open source)

What do we test?

- Geometry polygons, layers and cell names.
- Component Settings.
- Port positions and ensure they are on grid.

```python
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.difftest import difftest
from gdsfactory.samples.pdk.fab_c import cells

cell_names = list(cells.keys())
dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds")

@pytest.fixture(params=cell_names, scope="function")
def component_name(request) -> str:
    return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS names, shapes and layers.

    Runs XOR and computes the area.

    """
    component = cells[component_name]()
    test_name = f"fabc_{component_name}"
    difftest(component, test_name=test_name, dirpath=dirpath)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions in component settings and ports."""
    component = cells[component_name]()
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component_name: str) -> None:
    """Ensures all ports are on grid to avoid 1nm gaps."""
    component = cells[component_name]()
    component.assert_ports_on_grid()

```

For questions join the [![Join the chat at https://gitter.im/gdsfactory-dev/community](https://badges.gitter.im/gdsfactory-dev/community.svg)](https://gitter.im/gdsfactory-dev/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) with [element.io](https://element.io/download) or use GitHub issues or discussions.

## Running notebooks

You can find the tutorial jupyter notebooks in `notebooks` and open them with VSCode.

You can use [VSCode gdsfactory extension](https://marketplace.visualstudio.com/items?itemName=gdsfactory.gdsfactory) to open and run the notebooks.
