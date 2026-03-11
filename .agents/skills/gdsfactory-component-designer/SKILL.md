---
name: gdsfactory-component-designer
description: >
  Generate, visualize, and modify photonic integrated circuit components using
  gdsfactory. Use when the user asks to create, view, edit, or explore PIC
  components, waveguides, MMIs, ring resonators, gratings, or any layout from
  the gdsfactory component library or an activated PDK.
compatibility: Requires Python >=3.11, gdsfactory, and matplotlib.
metadata:
  author: gdsfactory
  version: "1.0"
allowed-tools: bash python
---

# gdsfactory Component Designer Skill

This skill lets an LLM agent **generate**, **visualize**, and **iteratively
modify** photonic-IC components using the
[gdsfactory](https://github.com/gdsfactory/gdsfactory) Python library.

---

## When to use this skill

Activate this skill when the user:

- Asks to create or generate a photonic component (waveguide, MMI, ring
  resonator, coupler, grating coupler, spiral, taper, MZI, etc.).
- Wants to see what a component looks like (requests a plot, image, or
  preview).
- Wants to tweak component parameters (width, length, radius, gaps, layers,
  cross-sections, etc.) and see the result.
- Mentions gdsfactory, GDS, photonic layout, or a specific PDK.
- Asks to compose multiple components together into a larger circuit.

---

## 1 — Setting up the environment

Before generating any component, make sure the PDK is activated.
The generic PDK ships with gdsfactory and is always available:

```python
import gdsfactory as gf

# Activate the built-in generic PDK (always available)
gf.gpdk.PDK.activate()
```

If the user specifies a third-party PDK (e.g. `cspdk`, `ubcpdk`,
`sky130`, `gf45spclo`), import and activate it instead:

```python
import cspdk

cspdk.PDK.activate()
```

Always clear the cell cache between independent component generations to avoid
stale state:

```python
gf.clear_cache()
```

---

## 2 — Generating a component

### 2.1 From the built-in component library

gdsfactory ships with 300+ parametric component factory functions under
`gf.components`. To instantiate a component, call its factory function:

```python
import gdsfactory as gf

gf.gpdk.PDK.activate()

# Example: 1×2 MMI splitter
c = gf.components.mmi1x2(width_mmi=5.0, length_mmi=25.0, gap_mmi=0.25)
```

You can also use the string-based lookup via the active PDK, which is useful
when the component name comes from user input:

```python
c = gf.get_component("mmi1x2", width_mmi=5.0, length_mmi=25.0)
```

### 2.2 Listing available components

To discover what components are available in the currently active PDK:

```python
pdk = gf.get_active_pdk()

# All cell factory names (sorted)
available = sorted(pdk.cells.keys())
print(available)
```

### 2.3 Inspecting component parameters

Every component factory function is a standard Python callable with typed
parameters. Use `help()` or `inspect.signature()` to discover the parameters:

```python
import inspect

sig = inspect.signature(gf.components.mmi1x2)
for name, param in sig.parameters.items():
    print(f"  {name}: {param.annotation} = {param.default}")
```

### 2.4 From a third-party PDK

When a PDK is activated its cells become available through `gf.get_component`:

```python
import cspdk

cspdk.PDK.activate()
c = gf.get_component("mmi1x2")  # uses cspdk's mmi1x2
```

---

## 3 — Visualizing a component (critical workflow)

Visualization is essential: always render and inspect the component after
creating or modifying it so you can verify the result and share it with the
user.

### 3.1 Save a plot image to disk and display it

Use the helper script bundled with this skill for reliable headless rendering.
From a bash tool or Python subprocess:

```bash
python .agents/skills/gdsfactory-component-designer/scripts/visualize_component.py \
    "gf.components.mmi1x2(width_mmi=5.0, length_mmi=25.0)" \
    /tmp/mmi1x2.png
```

Or do it inline in Python:

```python
import gdsfactory as gf
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt

gf.gpdk.PDK.activate()
c = gf.components.mmi1x2(width_mmi=5.0, length_mmi=25.0)

fig = c.plot(return_fig=True)
fig.savefig("/tmp/mmi1x2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Image saved to /tmp/mmi1x2.png")
```

After saving, **always import the image into context** so you and the user can
see it:

1. Use the screenshot/image-viewing tool on the saved file path.
2. Describe what you see in the plot (ports, shapes, layers, dimensions).
3. Ask the user if the result matches their expectations before moving on.

### 3.2 Quick component info

Print a textual summary before or alongside the image:

```python
print(f"Component: {c.name}")
print(f"Ports: {[p.name for p in c.ports]}")
print(f"Bounding box size: {c.dxsize:.3f} × {c.dysize:.3f} µm")
```

---

## 4 — Modifying a component

### 4.1 Adjusting parameters

The simplest modification is changing the factory-function arguments:

```python
gf.clear_cache()  # clear cache before regenerating

# Wider MMI with longer coupling region
c = gf.components.mmi1x2(width_mmi=8.0, length_mmi=40.0, gap_mmi=0.5)
```

Always clear the cache (`gf.clear_cache()`) before creating a component with
modified parameters to avoid returning a cached version with old values.

### 4.2 Composing components

Build a custom component by placing and connecting sub-components:

```python
@gf.cell
def my_circuit(mmi_length: float = 25.0) -> gf.Component:
    c = gf.Component()

    mmi = c.add_ref(gf.components.mmi1x2(length_mmi=mmi_length))
    bend = c.add_ref(gf.components.bend_euler(radius=10))

    # Connect bend input to one of the MMI outputs
    bend.connect("o1", mmi.ports["o2"])

    # Expose external ports
    c.add_port("o1", port=mmi.ports["o1"])
    c.add_port("o2", port=bend.ports["o2"])
    c.add_port("o3", port=mmi.ports["o3"])
    return c

gf.clear_cache()
c = my_circuit(mmi_length=30.0)
```

### 4.3 Editing geometry directly

For more advanced changes, add polygons or modify existing ones:

```python
@gf.cell
def custom_shape() -> gf.Component:
    c = gf.Component()
    # Add a rectangular polygon on the WG layer
    c.add_polygon(
        [(0, 0), (10, 0), (10, 0.5), (0, 0.5)],
        layer="WG",
    )
    c.add_port(name="o1", center=(0, 0.25), width=0.5,
               orientation=180, layer="WG")
    c.add_port(name="o2", center=(10, 0.25), width=0.5,
               orientation=0, layer="WG")
    return c
```

### 4.4 Using cross-sections

Cross-sections define the layer stack for a waveguide path:

```python
xs = gf.cross_section.strip(width=0.6)
c = gf.components.straight(length=20, cross_section=xs)
```

---

## 5 — Exporting the component

```python
# Write to GDS file
gdspath = c.write_gds("/tmp/my_component.gds")
print(f"GDS written to {gdspath}")
```

---

## 6 — Iterative design loop (recommended workflow)

When the user asks for a component, follow this loop:

1. **Clarify** what the user wants (component type, parameters, PDK).
2. **Generate** the component in Python.
3. **Visualize** — save a PNG and display it to the user.
4. **Describe** what you see (ports, dimensions, shapes, layers).
5. **Ask** if modifications are needed.
6. **Modify** parameters / code, clear cache, regenerate, and re-visualize.
7. **Export** the final GDS when the user is satisfied.

---

## 7 — Common component quick-reference

| Component | Factory function | Key parameters |
|---|---|---|
| Straight waveguide | `gf.components.straight` | `length`, `width`, `cross_section` |
| Euler bend | `gf.components.bend_euler` | `radius`, `angle`, `cross_section` |
| Circular bend | `gf.components.bend_circular` | `radius`, `angle` |
| S-bend | `gf.components.bend_s` | `size`, `cross_section` |
| 1×2 MMI | `gf.components.mmi1x2` | `width_mmi`, `length_mmi`, `gap_mmi` |
| 2×2 MMI | `gf.components.mmi2x2` | `width_mmi`, `length_mmi`, `gap_mmi` |
| Ring resonator | `gf.components.ring_single` | `gap`, `radius`, `length_x`, `length_y` |
| Double ring | `gf.components.ring_double` | `gap`, `radius`, `length_x`, `length_y` |
| Directional coupler | `gf.components.coupler` | `gap`, `length`, `dy` |
| Mach-Zehnder | `gf.components.mzi` | `delta_length`, `length_x`, `length_y` |
| Grating coupler (TE) | `gf.components.grating_coupler_te` | `period`, `n_periods`, `taper_length` |
| Taper | `gf.components.taper` | `length`, `width1`, `width2` |
| Spiral | `gf.components.spiral` | `length`, `N`, `spacing` |
| Pad | `gf.components.pad` | `size`, `layer` |
| Via stack | `gf.components.via_stack` | `size`, `layers`, `vias` |

---

## 8 — Troubleshooting

| Problem | Fix |
|---|---|
| `ValueError: No active PDK` | Call `gf.gpdk.PDK.activate()` first. |
| Component returns cached version with old params | Call `gf.clear_cache()` before creating the component. |
| `ModuleNotFoundError` for a PDK | Install it: `pip install <pdk-package>`. |
| Blank or black plot image | Ensure `matplotlib.use("Agg")` is set before importing `pyplot`. |
| Port connection error | Check port names with `c.ports` and ensure widths/layers match. |
