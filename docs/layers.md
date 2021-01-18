# Layers

Each foundry uses different GDS numbers for each process step.

We follow the generic layer numbers from the book "Silicon Photonics Design: From Devices to Systems Lukas Chrostowski, Michael Hochberg". See `pp.LAYER`

| GDS (layer, purpose) | layer_name | Description                                                 |
| -------------------- | ---------- | ----------------------------------------------------------- |
| 1 , 0                | WG         | 220 nm Silicon core                                         |
| 2 , 0                | SLAB150    | 150nm Silicon slab (70nm shallow Etch for grating couplers) |
| 3 , 0                | SLAB90     | 90nm Silicon slab (for modulators)                          |
| 47, 0                | MH         | heater                                                      |
| 41, 0                | M1         | metal 1                                                     |
| 45, 0                | M2         | metal 2                                                     |
| 40, 0                | VIA1       | VIA1                                                        |
| 44, 0                | VIA2       | VIA2                                                        |
| 46, 0                | PADOPEN    | Bond pad opening                                            |
| 51, 0                | UNDERCUT   | Undercut                                                    |
| 52, 0                | DEEPTRENCH | Deep trench                                                 |
| 66, 0                | TEXT       | Text markup                                                 |
| 64, 0                | FLOORPLAN  | Mask floorplan                                              |

Layers are available in `pp.LAYER` as `pp.LAYER.WG`, `pp.LAYER.WGCLAD`

You can build PDKs for different foundries using gdsfactory, the PDKs contain some foundry IP such as layer numbers, minimum CD, layer stack, so you need to keep them in a separate private repo. See [UBC PDK](https://github.com/gdsfactory/ubc) as an example.

I reccommend that you create the PDK repo using a cookiecutter template. For example, you can use this one.

```
pip install cookiecutter
cookiecutter https://github.com/joamatab/cookiecutter-pypackage-minimal
```

Or you can fork the UBC PDK and create new cell functions that use the correct layers for your foundry. For example.

```

from dataclasses import dataclass
import pp


@dataclass
class Layer:
    WGCORE = (3, 0)
    LABEL = (100, 0)


LAYER = Layer()


@pp.cell
def waveguide(length=10, width=0.5):
    return pp.c.waveguide(length=length, width=width, layer=LAYER.WGCORE, layers_cladding=[])

```
