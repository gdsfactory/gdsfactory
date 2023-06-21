# Installation for developers

As a contributor, if you are on windows you need to download [Git](https://git-scm.com/download/win) and optionally [GitHub Desktop](https://desktop.github.com/).

Then you need to fork the [GitHub repository](https://github.com/gdsfactory/gdsfactory), git clone it (download it), git add, git commit, git push your improvement. Then pull request your changes to the main branch from the GitHub website.
For that you can install gdsfactory locally on your computer in `-e` edit mode.

```
git clone git@github.com:YourUserName/gdsfactory.git
cd gdsfactory
git clone https://github.com/gdsfactory/gdsfactory-test-data.git -b test-data test-data
mamba install gdstk -y
pip install -e .[full] pre-commit
pre-commit install
```

## Style

- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and take a look at existing gdsfactory code.
- Make sure tests pass on GitHub.
- Install pre-commit to get the pre-commit checks passing (autoformat the code, run linter ...).

```
cd gdsfactory
pip install -e . pre-commit
pre-commit install
```

Pre-commit makes sure your code is formatted following black and checks syntax.
If you forgot to `pre-commit install` you can fix pre-commit issues by running

```
pre-commit run --all-files
```

## Tests

`pytest` runs 3 types of tests:

You can run tests with `pytest`. This will run 3 types of tests:

- pytest will test any function that starts with `test_`. You can assert the number of polygons, the name, the length of a route or whatever you want.
- regressions tests: avoids unwanted regressions by storing Components port locations in CSV and metadata in YAML files. You can force to regenerate the reference files running `pytest --force-regen -s` from the repo root directory.
  - `tests/test_containers.py` stores container settings in YAML and port locations in a CSV file
  - `tests/components/test_components.py` stores all the component settings in YAML
  - `tests/components/test_ports.py` stores all port locations in a CSV file
  - `tests/test_netlists.py` stores all the component netlist in YAML and rebuilds the component from the netlist.
    - converts the routed PIC into YAML and build back into the same PIC from its YAML definition
  - lytest: writes all components GDS in `run_layouts` and compares them with `ref_layouts`
    - when running the test it will do a boolean of the `run_layout` and the `ref_layout` and raise an error for any significant differences.
    - you can check out any changes in your library with `gf gds diff ref_layouts/bbox.gds run_layouts/bbox.gds`

If test failed because you modified the geometry you can regenerate the regression tests with:

```
pytest --force-regen -s
```

## Build your own Reticles/projects/PDKs

We recommend creating a separate python project for each mask and PDK.

```
pip install cookiecutter
cookiecutter https://github.com/joamatab/cookiecutter-pypackage-minimal

```

## Test own Reticles/projects/PDKs

You can take a look at other open source PDKs tests code.

- [GlobalFoundries 180nm MCU CMOS PDK](https://gdsfactory.github.io/gf180/) (open source)
- [SiEPIC Ebeam UBC PDK](https://gdsfactory.github.io/ubc) (open source)
- [Skywater130 CMOS PDK](https://gdsfactory.github.io/skywater130) (open source)

What do we test?

- Geometry polygons, layers and cell names.
- Component Settings.
- Port positions and ensure they are on grid.


```python
import pathlib
import pytest
from pytest_regressions.data_regression import DataRegressionFixture

from gdsfactory.component import Component
from gdsfactory.difftest import difftest
from ubcpdk import cells


skip_test = { }
cell_names = set(cells.keys()) - set(skip_test)
dirpath_ref = pathlib.Path(__file__).absolute().parent / "ref"


@pytest.fixture(params=cell_names, scope="function")
def component(request) -> Component:
    return cells[request.param]()


def test_pdk_gds(component: Component) -> None:
    """Avoid regressions in GDS geometry, cell names and layers."""
    difftest(component, dirpath=dirpath_ref)


def test_pdk_settings(
    component: Component, data_regression: DataRegressionFixture
) -> None:
    """Avoid regressions when exporting settings."""
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component: Component):
    component.assert_ports_on_grid()

```

- For questions you can join the chat[![Join the chat at https://gitter.im/gdsfactory-dev/community](https://badges.gitter.im/gdsfactory-dev/community.svg)](https://gitter.im/gdsfactory-dev/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) with [element.io](https://element.io/download) or any matrix client.
- Also create GitHub issues or use GitHub discussions.
