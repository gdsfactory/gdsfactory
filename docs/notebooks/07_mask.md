# Die and mask assembly

With gdsfactory you can easily go from a simple component, to a component with many components inside.

In the same way that you need to Layout for DRC (Design Rule Check) clean devices, you have to layout obeying the Design for Test (DFT) and Design for Packaging rules.

## Design for Test

To measure your chips after fabrication you need to decide your test configurations. This includes Design For Testing Rules like:

- `Individual input and output fibers` versus `fiber array`. You can use `add_fiber_array` for easier testing and higher throughput, or `add_fiber_single` for the flexibility of single fibers.
- Fiber array pitch (127um or 250um) if using a fiber array.
- Pad pitch for DC and RF high speed probes (100, 125, 150, 200um). Probe configuration (GSG, GS ...)
- Test layout for DC, RF and optical fibers.

```python
from functools import partial

import gdsfactory as gf



gf.gpdk.PDK.activate()
# After you run gf.config.rich_output(), gdsfactory will automatically generate and display a plot of the component's geometry instead of the text.
gf.config.rich_output()

```

## Pack

Let us start with a resistance sweep, where you change the resistance width to measure sheet resistance.

```python
def add_resistance_sweep_info(c):
    # A helper function is defined to add specific metadata to a component's .info attribute.
    # This information, like "doe": "resistance_sweep" (Design of Experiments) and "measurement": "iv" (current-voltage),
    # is used by automated testing and data analysis tools.
    c.info["doe"] = "resistance_sweep"
    c.info["analysis"] = "[iv_resistance]"
    c.info["analysis_parameters"] = "[{}]"
    c.info["ports_electrical"] = 2
    c.info["ports_optical"] = 0
    c.info["measurement"] = "iv"
    c.info["measurement_parameters"] = "{}"
    return c

# A list of three resistance_sheet components is created, each with a different width (1, 10, and 50 µm).
sweep = [gf.components.resistance_sheet(width=width) for width in [1, 10, 50]]

# The add_resistance_sweep_info function is applied to each of the three resistor components, embedding the testing metadata into each one.
sweep_with_info = [add_resistance_sweep_info(c) for c in sweep]

# The gf.pack function takes the list of three components and automatically arranges them in a compact way,
# creating a single, larger component that contains them. m[0] retrieves this final packed component.
m = gf.pack(sweep_with_info)
c = m[0]
c.draw_ports()
c.pprint_ports()
c.plot()
```

```python
sweep_with_info[0].info
```

Then we add spirals with different lengths to measure the waveguide propagation loss. You can use both a fiber array or single fiber.

```python
@gf.cell

# This line creates the basic spiral waveguide. The **kwargs allows you to pass any parameters (like length, width, etc.) to the underlying spiral component.
def spiral_gc(**kwargs):
    """Returns spiral with Grating Couplers."""
    c = gf.components.spiral(**kwargs)
    c = gf.routing.add_fiber_array(c)
    c.info["doe"] = "spirals_sc"  # Strip Cband spirals
    c.info["measurement"] = "optical_spectrum"
    c.info["measurement_parameters"] = "{}"
    c.info["analysis"] = "[power_envelope]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_optical"] = 4
    c.info["ports_electrical"] = 0
    c.info.update(kwargs)
    return c


c = spiral_gc(length=100)
c.plot()
```

```python
c.info
```

```python
sweep = [spiral_gc(length=length) for length in [100, 200, 300]]
m = gf.pack(sweep)
c = m[0]
c.plot()
```

You can also add some physical labels that will be fabricated.
For example you can add prefix `S` at the `north-center` of each spiral using `text_rectangular` which is DRC clean and anchored on `nc` (north-center).

```python
text_metal = partial(gf.components.text_rectangular_multi_layer, layers=("M1",))

m = gf.pack(sweep, text=text_metal, text_anchors=("cw",), text_prefix="s")
c = m[0]
c.show()
c.plot()
```

## Grid

You can also pack components with a constant spacing.

```python
g = gf.grid(sweep)
g.plot()
```

```python
gh = gf.grid(sweep, shape=(1, len(sweep)))
gh.plot()
```

```python
gh_ymin = gf.grid(sweep, shape=(len(sweep), 1), align_x="xmin")
gh_ymin.plot()
```

Additionally, it allows you to add text labels to each element of the sweep

```python
gh_ymin = gf.grid_with_text(
    
    # shape=(len(sweep), 1): This defines the grid as having a single column and a number of rows equal to the number of components in the sweep list.
    # align_x="xmax": This aligns all the components in the column so that their right edges (xmax) are in a straight vertical line.
    # text=text_metal: This provides a list of text strings that will be placed as labels next to each corresponding component in the grid.
    sweep, shape=(len(sweep), 1), align_x="xmax", text=text_metal
)
gh_ymin.plot()
```

```python
gh_ymin = gf.grid_with_text(
    sweep,
    shape=(len(sweep), 1),
    align_x="xmax",
    text=text_metal,
    labels=("S100", "S200", "S300"),
)
gh_ymin.plot()
```

You have 2 ways of defining a mask:

1. in YAML
2. in Python

## YAML Component

You can also define your Component in the YAML format thanks to `gdsfactory.read.from_yaml`

You need to define:

- Instances
- Placements
- Routes (optional)

And you can leverage:

1. `pack_doe`
2. `pack_doe_grid`


`pack_doe` places components as compact as possible.


`pack_doe_grid` places each component on a regular grid.

```python

c = gf.read.from_yaml(
    """
name: mask_compact

instances:
  rings:

  #  This is a special function that creates a Design of Experiments array.
    component: pack_doe
    settings:
      
      # settings: radius: [30, 50, 20, 40], length_x: [1, 2, 3]: It will create ring resonators with these different radii and coupling lengths.
      doe: ring_single
      settings:
        radius: [30, 50, 20, 40]
        length_x: [1, 2, 3]

      # This tells the function to generate all possible combinations of the specified radius and length_x values.
      do_permutations: True
      function:

        # After each unique ring is created, the add_fiber_array function is applied to it, adding grating couplers for testing.
        function: add_fiber_array
        settings:
            fanout_length: 200


  mzis:
    component: pack_doe_grid
    settings:
      doe: mzi
      settings:
        delta_length: [10, 100]
      do_permutations: True
      spacing: [10, 10]
      function: add_fiber_array

placements:
  rings:
    xmin: 50

  mzis:
    xmin: rings,east
"""
)

c.show()
c.plot()
```


## Automated testing and analysis 

This is useful when you have a lot of components and you want to automate the testing process.

There are two main ways to define which components are testable:

1. Include a `doe` (Design of Experiments) field in the component.info dictionary, as well as all relevant test and analysis information.
2. Include a GDS label in all component test points. There are many ways to define test points, but the most common is to use a GDS label with the format `<elec/opt>-<number_of_ports>-<cell_name>`. This way you can easily extract all test points from the GDS file.

```python
import pandas as pd

import gdsfactory as gf


@gf.cell
def mzm_gc(length_x=10, **kwargs) -> gf.Component:
    """Returns a MZI with Grating Couplers.

    Args:
        length_x: length of the MZI.
        kwargs: additional settings.
    """
    c = gf.components.mzi2x2_2x2_phase_shifter(
        length_x=length_x, auto_rename_ports=False, **kwargs
    )
    c = gf.routing.add_pads_top(c, port_names=["top_l_e1", "top_r_e3"])
    c = gf.routing.add_fiber_array(c)
    c.info["doe"] = "mzm"
    c.info["measurement"] = "optical_spectrum"
    c.info["analysis"] = "[fsr]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_electrical"] = 2
    c.info["ports_optical"] = 6
    c.info["length_x"] = length_x
    c.info.update(kwargs)
    return c


def sample_reticle(grid: bool = False) -> gf.Component:
    """Returns MZI with TE grating couplers."""

    mzis = [mzm_gc(length_x=lengths) for lengths in [100, 200, 300]]
    spirals = [spiral_gc(length=length) for length in [0, 100, 200]]
    rings = []
    for length_x in [10, 20, 30]:
        ring = gf.components.ring_single_heater(length_x=length_x)
        c = gf.components.add_fiber_array_optical_south_electrical_north(
            component=ring,
            electrical_port_names=["l_e2", "r_e2"],
            grating_coupler=gf.components.grating_coupler_te, 
            pad=gf.components.pad,
            cross_section_metal='metal3'
        ).copy()
        c.name = f"ring_{length_x}"
        c.info["doe"] = "ring_length_x"
        c.info["measurement"] = "optical_spectrum"
        c.info["ports_electrical"] = 2
        c.info["ports_optical"] = 4
        c.info["analysis"] = "[fsr]"
        c.info["analysis_parameters"] = "[]"
        c.info["length_x"] = length_x
        rings.append(c)

    copies = 3  # Number of copies of each component.
    components = mzis * copies + rings * copies + spirals * copies
    if grid:
        return gf.grid(components)
    mask = gf.pack(components)
    if len(mask) > 1:
        mask = gf.pack(mask)
    return mask[0]


c = sample_reticle()
c.show()
c
```

```python
gf.labels.write_test_manifest(c, csvpath="sample_reticle.csv")
df = pd.read_csv("sample_reticle.csv")
df
```

You can see a test manifest example [here](https://docs.google.com/spreadsheets/d/1845m-XZM8tZ1tNd8GIvAaq7ZE-iha00XNWa0XrEOabc/edit#gid=233591479)

## Automated testing with labels

The GDS info is stored in the GDS file metadata and can be lost if the GDS file is modified with other tools that are not aware of the metadata. To avoid this, GDSFactory also supports a more traditional way of defining test points, using GDS labels. 

For example, lets say you want to label the rightmost port of a component with a GDS label `port_type-number_of_ports-cell_name`. You can do this with the following code:


```python
import gdsfactory as gf
from gdsfactory.typings import LayerSpec

layer_label = "TEXT"


def label_farthest_right_port(
    component: gf.Component, ports: gf.Port | list[gf.Port], layer: LayerSpec, text: str
) -> gf.Component:
    """Adds a label to the right of the farthest right port in a given component.

    Args:
        component: The component to which the label is added.
        ports: A list of ports to evaluate for positioning the label.
        layer: The layer on which the label will be added.
        text: The text to display in the label.
    """
    rightmost_port = max(ports, key=lambda port: port.dx)

    component.add_label(
        text=text,
        position=rightmost_port.dcenter,
        layer=layer,
    )
    return component


c = gf.Component()
ref = c << gf.routing.add_pads_top(gf.components.wire_straight())

# ref.ports: The function takes the list of all ports from the component reference ref.
# It then iterates through these ports to find the one with the largest x-coordinate (the one that is farthest to the right).
# c.add_label(...): Once the rightmost port is identified, the function adds a text label to the main component c at that port's position.
label_farthest_right_port(c, ref.ports, layer=layer_label, text="elec-2-wire_straight")
c
```

```python
def spiral_gc(length: float = 0, **kwargs) -> gf.Component:
    """Returns a spiral double with Grating Couplers.

    Args:
        length: length of the spiral straight section.
        kwargs: additional settings.

    Keyword Args:
        bend: bend component.
        straight: straight component.
        cross_section: cross_section component.
        spacing: spacing between the spiral loops.
        n_loops: number of loops.
    """
    c0 = gf.c.spiral(length=length, **kwargs)
    c = gf.routing.add_fiber_array(c0)
    c.info["doe"] = "spirals_sc"
    c.info["measurement"] = "optical_spectrum"
    c.info["analysis"] = "[power_envelope]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_optical"] = 4
    c.info["ports_electrical"] = 0
    c.info.update(kwargs)

    c.name = f"spiral_gc_{length}"
    label_farthest_right_port(c, c.ports, layer=layer_label, text=f"opt-4-{c.name}")
    return c


c = spiral_gc(length=0)
c
```

```python
def mzi_gc(length_x=10, **kwargs) -> gf.Component:
    """Returns a MZI with Grating Couplers.

    Args:
        length_x: length of the MZI.
        kwargs: additional settings.
    """
    c = gf.components.mzi2x2_2x2_phase_shifter(
        length_x=length_x, auto_rename_ports=False, **kwargs
    )
    c = gf.routing.add_pads_top(c, port_names=("top_l_e1", "top_r_e3"))
    c.name = f"mzi_{length_x}"
    c = gf.routing.add_fiber_array(c)

    c.info["doe"] = "mzi"
    c.info["measurement"] = "optical_spectrum"
    c.info["analysis"] = "[fsr]"
    c.info["analysis_parameters"] = "[]"
    c.info["ports_electrical"] = 2
    c.info["ports_optical"] = 6
    c.info["length_x"] = length_x
    c.info.update(kwargs)

    c.name = f"mzi_gc_{length_x}"
    label_farthest_right_port(
        c,
        c.ports.filter(port_type="vertical_te"),
        layer=layer_label,
        text=f"opt-{c.info['ports_optical']}-{c.name}",
    )
    label_farthest_right_port(
        c,
        c.ports.filter(port_type="electrical"),
        layer=layer_label,
        text=f"elec-{c.info['ports_electrical']}-{c.name}",
    )
    return c


c = mzi_gc(length_x=10)
c
```

```python
def sample_reticle_with_labels(grid: bool = False) -> gf.Component:
    """Returns MZI with TE grating couplers."""

    mzis = [mzi_gc(length_x=lengths) for lengths in [100, 200, 300]]
    spirals = [spiral_gc(length=length) for length in [0, 100, 200]]
    rings = []
    for length_x in [10, 20, 30]:
        ring = gf.components.ring_single_heater(length_x=length_x)
        c = gf.components.add_fiber_array_optical_south_electrical_north(
            component=ring,
            electrical_port_names=["l_e2", "r_e2"],
            grating_coupler=gf.components.grating_coupler_te, 
            pad=gf.components.pad,
            cross_section_metal='metal3'
        ).copy()
        c.name = f"ring_{length_x}"
        c.info["doe"] = "ring_length_x"
        c.info["measurement"] = "optical_spectrum"
        c.info["ports_electrical"] = 2
        c.info["ports_optical"] = 4
        c.info["analysis"] = "[fsr]"
        c.info["analysis_parameters"] = "[]"
        label_farthest_right_port(
            c,
            c.ports.filter(port_type="vertical_te"),
            layer=layer_label,
            text=f"opt-{c.info['ports_optical']}-{c.name}",
        )
        label_farthest_right_port(
            c,
            c.ports.filter(port_type="electrical"),
            layer=layer_label,
            text=f"elec-{c.info['ports_electrical']}-{c.name}",
        )
        rings.append(c)

    copies = 3  # Number of copies of each component.
    components = mzis * copies + rings * copies + spirals * copies

    # gf.grid(components): This function is called to arrange the components in a simple, evenly spaced grid.
    if grid:
        return gf.grid(components)
    
    # gf.pack(components): This is the default path. The pack function uses a more advanced algorithm to arrange the components in a space-efficient way,
    # which is important for minimizing the cost of fabrication. This may result in a list of one or more packed groups.
    c = gf.pack(components)
    
    # The pack function can sometimes return multiple packed groups. This code checks if that is the case (len(c) > 1) and, if so,
    # runs pack again on those groups to combine them into one final, single component. The [0] then selects that final component.
    if len(c) > 1:
        c = gf.pack(c)[0]
    return c[0]


c = sample_reticle_with_labels()
c
```

You can also extract all test points from a GDS file using gf.labels.write_labels

```python
import pandas as pd

gdspath = c.write_gds()
csvpath = gf.labels.write_labels(gdspath, layer_label=layer_label)
df = pd.read_csv(csvpath)
df = df.sort_values(by=["text"])
df
```
