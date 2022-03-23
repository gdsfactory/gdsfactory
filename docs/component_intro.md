# Workflow

you'll need to keep 3 windows open:

1. A text editor or IDE (Visual Studio Code, Pycharm, Spyder, neovim, Atom, Jupyterlab ...)
2. A python / Ipython terminal / jupyter notebook (interactive python to run).
3. Klayout to Visualize the GDS files.

`Component.show()` will stream the GDS to klayout so klayout needs to be open.
Make sure you also ran `gf tool install` from the terminal to install the `gdsfactory` to `klayout` interface.


![windows](https://i.imgur.com/inzGBb5.png)


## Layers

Each foundry uses different GDS numbers for each process step.

We follow the generic layer numbers from the book "Silicon Photonics Design: From Devices to Systems Lukas Chrostowski, Michael Hochberg".

| GDS (layer, purpose) | layer_name | Description                                                 |
| -------------------- | ---------- | ----------------------------------------------------------- |
| 1 , 0                | WG         | 220 nm Silicon core                                         |
| 2 , 0                | SLAB150    | 150nm Silicon slab (70nm shallow Etch for grating couplers) |
| 3 , 0                | SLAB90     | 90nm Silicon slab (for modulators)                          |
| 4, 0                 | DEEPTRENCH | Deep trench                                                 |
| 47, 0                | MH         | heater                                                      |
| 41, 0                | M1         | metal 1                                                     |
| 45, 0                | M2         | metal 2                                                     |
| 40, 0                | VIAC       | VIAC to contact Ge, NPP or PPP                              |
| 44, 0                | VIA1       | VIA1                                                        |
| 46, 0                | PADOPEN    | Bond pad opening                                            |
| 51, 0                | UNDERCUT   | Undercut                                                    |
| 66, 0                | TEXT       | Text markup                                                 |
| 64, 0                | FLOORPLAN  | Mask floorplan                                              |

Layers are available in `gf.LAYER` as `gf.LAYER.WG`, `gf.LAYER.WGCLAD`

You can build PDKs for different foundries. The PDKs contain some foundry IP such as layer numbers, minimum CD, layer stack, so you need to keep them in a separate private repo. See [UBC PDK](https://github.com/gdsfactory/ubc) as an example.

I recommend that you create the PDK repo using a cookiecutter template. For example, you can use this one.

```
pip install cookiecutter
cookiecutter https://github.com/joamatab/cookiecutter-pypackage-minimal
```

Or you can fork the UBC PDK and create new cell functions that use the correct layers for your foundry. For example.

```

import dataclasses
import gdsfactory as gf


@dataclasses.dataclass(frozen=True)
class Layer:
    WGCORE = (3, 0)
    LABEL = (100, 0)


LAYER = Layer()


```

## Types

What are the common data types?

```{eval-rst}
.. automodule:: gdsfactory.types
```


## Tests

As you write your own component factories you want to make sure you do not break them later on.
The best way of doing that is writing tests to avoid unwanted regressions.

Here is an example on how to group your functions in a dict.

Make sure you create a test like this an name it with `test_` prefix so Pytest can find it.

```python
import pathlib

import pytest
from pytest_regressions.data_regression import DataRegressionFixture

import gdsfactory as gf
from gdsfactory.difftest import difftest

dirpath = pathlib.Path(__file__).absolute().with_suffix(".gds")


mmi_long = gf.partial(gf.components.mmi, length_mmi=40)
mmi_short = gf.partial(gf.components.mmi, length_mmi=20)

factory = dict(
    mmi_short=mmi_short,
    mmi_long=mmi_long,
)


component_names = list(factory.keys())


@pytest.fixture(params=component_names, scope="function")
def component_name(request) -> str:
    return request.param


def test_gds(component_name: str) -> None:
    """Avoid regressions in GDS names, shapes and layers.
    Runs XOR and computes the area."""
    component = factory[component_name]()
    test_name = f"fabc_{component_name}"
    difftest(component, test_name=test_name, dirpath=dirpath)


def test_settings(component_name: str, data_regression: DataRegressionFixture) -> None:
    """Avoid regressions in component settings and ports."""
    component = factory[component_name]()
    data_regression.check(component.to_dict())


def test_assert_ports_on_grid(component_name: str):
    """Ensures all ports are on grid to avoid 1nm gaps"""
    component = factory[component_name]()
    component.assert_ports_on_grid()
```
