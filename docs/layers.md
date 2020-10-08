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

You can build PDKs for different foundries using gdsfactory, the PDKs contain some foundry IP such as layer numbers, minimum CD, layer stack, so you need to keep them in a separate private repo.

You basically have two options:

1. You create a PDK that has all the layer information for the foundry.

2. You use the generic layermap from the table and use a script to remap the GDS layers into the specific foundry layer map, using for example a klayout script or using the remap shows in `pp/samples/06_remaping_layers.py`

```

import pp
from fab.layers import layer

def waveguide(length=10, width=0.5):
    return pp.c.waveguide(length=length, width=width, layer=layer.WGCORE, layers_cladding=[layer.WGCLAD])

c = waveguide()
```
