# Workflow

You'll need 2 windows:

1. A text editor or IDE (Visual Studio Code, Pycharm, Spyder, neovim, Atom, Jupyterlab ...)
2. Klayout to Visualize the GDS files.

`Component.show()` will stream the GDS to klayout so klayout needs to be open.
Make sure you also ran `gf tool install` from the terminal to install the `gdsfactory` to `klayout` interface.


## 1. Python Driven flow

1. You write your Pcells in python.
2. You execute the python code.
3. You visualize the GDS Layout in Klayout.

![windows](https://i.imgur.com/ZHEAotn.png)



## 2. YAML driven flow

For building masks and complex circuits you can also use the `netlist` driven flow.

The best way to run the netlist driven flow is using a file watcher, where you keep a live process that updates your GDS file live.


1. You write your netlist in YAML. It's basically a Place and Auto-Route.
2. You execute the file watcher `gf yaml watch FolderName`.
3. You visualize the GDS Layout in Klayout.

![yaml](https://i.imgur.com/h1ABhJ9.png)

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

from pydantic import BaseModel
import gdsfactory as gf


class LayerMap(BaseModel):
    WGCORE = (3, 0)
    LABEL = (100, 0)
    DEVREC: Layer = (68, 0)
    LABEL: Layer = (10, 0)
    PORT: Layer = (1, 10)  # PinRec
    PORTE: Layer = (1, 11)  # PinRecM
    FLOORPLAN: Layer = (99, 0)

    TE: Layer = (203, 0)
    TM: Layer = (204, 0)
    TEXT: Layer = (66, 0)
    LABEL_INSTANCE: Layer = (66, 0)


LAYER = LayerMap()


```

## Types

What are the common data types?

```{eval-rst}
.. automodule:: gdsfactory.types
```
