# EME with MEOW

Some components are more efficiently modeled with Eigenmode Expansion.

Gdsfactory provides a plugin for MEOW to efficiently extract component S-parameters through EME.

Currently the component needs to specifically have a single "o1" port facing west, and a single "o2" port facing east, like this taper:

```python
import matplotlib.pyplot as plt
from gdsfactory.simulation.eme import MEOW
import numpy as np
import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

c = gf.components.taper_cross_section_sine()
c
```

You also need to explicitly provide a LayerStack to define cross-sections, for instance the generic one:

```python
layerstack = gf.generic_tech.LAYER_STACK

filtered_layerstack = gf.technology.LayerStack(
    layers={
        k: layerstack.layers[k]
        for k in (
            "slab90",
            "core",
            "box",
            "clad",
        )
    }
)
```

Since you need to make sure that your entire LayerStack has e.g. material information for all present layers, it is safer to only keep the layers that you need for your simulation:


The EME simulator can be instantiated with only these two elements, alongside parameters:

```python
eme = MEOW(component=c, layerstack=filtered_layerstack, wavelength=1.55)
```

Plotting functions allow you to check your simulation:

```python
eme.plot_structure()
```

The cross-section themselves:

```python
eme.plot_cross_section(xs_num=0)
```

```python
eme.plot_cross_section(xs_num=-1)
```

And the modes (after calculating them):

```python
eme.plot_mode(xs_num=0, mode_num=0)
```

```python
eme.plot_mode(xs_num=-1, mode_num=0)
```

The S-parameters can be calculated, and are returned in the same format as for the FDTD solvers (the original MEOW S-parameter results S and port_names are saved as attributes):

```python
sp = eme.compute_sparameters()
```

```python
print(np.abs(sp["o1@0,o2@0"]) ** 2)
```

```python
print(eme.port_map)
eme.plot_Sparams()
```

As you can see most light stays on the fundamental TE mode


## Sweep EME length

Lets sweep the length of the taper.

```python
layerstack = gf.generic_tech.LAYER_STACK

filtered_layerstack = gf.technology.LayerStack(
    layers={
        k: layerstack.layers[k]
        for k in (
            "slab90",
            "core",
            "box",
            "clad",
        )
    }
)

c = gf.components.taper(width2=2)
c
```

Lets do a convergence tests on the `cell_length` parameter. This depends a lot on the structure.

<!-- #region -->
```python
import matplotlib.pyplot as plt

trans = []
cells_lengths = [0.1, 0.25, 0.5, 0.75, 1]

for cell_length in cells_lengths:
    m = MEOW(
        component=c,
        layerstack=filtered_layerstack,
        wavelength=1.55,
        resolution_x=100,
        resolution_y=100,
        spacing_x=1,
        spacing_y=-3,
        num_modes=4,
        cell_length=cell_length,
    )
    sp = m.compute_sparameters()
    te0_trans = np.abs(sp["o1@0,o2@0"]) ** 2
    trans.append(te0_trans)

plt.plot(cells_lengths, trans, ".-")
plt.title("10um taper, resx = resy = 100, num_modes = 4")
plt.xlabel("Cell length (um)")
plt.ylabel("TE0 transmission")
```

![](https://i.imgur.com/70dU6fo.png)
<!-- #endregion -->

```python
eme = MEOW(component=c, layerstack=filtered_layerstack, wavelength=1.55)
```

```python
eme.plot_cross_section(xs_num=0)
```

```python
eme.plot_mode(xs_num=0, mode_num=0)
```

```python
eme.plot_cross_section(xs_num=-1)
```

```python
eme.plot_mode(xs_num=-1, mode_num=0)
```

```python
sp = eme.compute_sparameters()
```

```python
print(eme.port_map)
eme.plot_Sparams()
```

```python
T = np.abs(sp["o1@0,o2@0"]) ** 2
T
```

```python
np.abs(sp["o1@0,o2@2"]) ** 2
```

```python
lengths = np.array([1, 2, 3])
T = np.zeros_like(lengths, dtype=float)
```

```python
for length in lengths:
    c = gf.components.taper(width2=2, length=length)
    c.plot()
```

```python
for i, length in enumerate(lengths):
    c = gf.components.taper(width2=10, length=length)
    eme = MEOW(
        component=c, layerstack=filtered_layerstack, wavelength=1.55, cell_length=0.4
    )
    sp = eme.compute_sparameters()
    T[i] = np.abs(sp["o1@0,o2@0"]) ** 2
```

```python
plt.plot(lengths, T, ".")
plt.title("Fundamental mode transmission")
plt.ylabel("Transmission")
plt.xlabel("taper length (um)")
```

```python
T
```

```python

```
