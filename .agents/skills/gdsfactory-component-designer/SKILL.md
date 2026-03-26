---
name: gdsfactory-component-designer
description: >
  Expert photonic IC designer using gdsfactory. ALWAYS use this skill for ANY
  task involving silicon photonics, photonic integrated circuits (PICs), GDS
  layout design, or component visualization. Trigger this skill whenever the
  user mentions photonic components (waveguides, MMIs, rings, gratings, MZIs),
  layout design, or asks to "design", "generate", "visualize", "tweak", or
  "assemble" photonic structures. Do not wait for an explicit mention of
  "gdsfactory" – if the task involves photonic layout or PICs, this is the
  required tool.
compatibility: Requires Python >=3.11, gdsfactory, and matplotlib.
metadata:
  author: gdsfactory
  version: "1.1"
allowed-tools: bash python uv
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
- Mentions GDS, photonic layout, or a specific PDK.
- Asks to compose multiple components together into a larger circuit.

---

## 1 — Setting up the environment

### Choosing the right Python command

The examples below use bare `python`, but you must adapt the invocation to
whatever Python environment the user has set up. Common alternatives:

| Setup | Command |
|---|---|
| System / venv / conda | `python` |
| uv project | `uv run python` |
| pipx-installed gdsfactory | `pipx run --spec gdsfactory python` |
| Nix shell | `nix develop -c python` |

Probe the environment first (e.g. check for a `pyproject.toml` with
`[tool.uv]`, or an active virtualenv) and pick the appropriate command.  When
in doubt, try `python -c "import gdsfactory"` — if it fails, fall back to
`uv run python` or ask the user.

### Activating the PDK

Before generating any component, make sure the PDK is activated.
The generic PDK ships with gdsfactory and is always available:

```python
import gdsfactory as gf

# Activate the built-in generic PDK (always available)
gf.gpdk.PDK.activate()
```

If the user specifies a third-party PDK (e.g. `cspdk`, `ubcpdk`,
`sky130`, `gf45spclo`), import and activate it instead.

Always clear the cell cache between independent component generations to avoid
stale state: `gf.clear_cache()`.

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

### 2.2 Inspecting component parameters

Every component factory function is a standard Python callable with typed
parameters. Use `help()` or `inspect.signature()` to discover the parameters.

---

## 3 — Visualizing a component (Proactive Workflow)

Visualization is essential: render and inspect the component after creating or
modifying it to verify the result.

### 3.1 Save a plot image to disk and display it

Use the helper script bundled with this skill for reliable headless rendering.
From a bash tool or Python subprocess:

```bash
python .agents/skills/gdsfactory-component-designer/scripts/visualize_component.py \
    "gf.components.mmi1x2(width_mmi=5.0, length_mmi=25.0)" \
    /tmp/mmi1x2.png
```

After saving, **always import the image into context** so you and the user can
see it.

- **Be Proactive:** If the user's intent is clear (e.g., "Create an MMI then
  change its width"), do not stop for permission after the first step. Perform
  the generation, visualization, and modification in a single turn if possible.
- **Concise Reporting:** Describe the component briefly (ports, size). If the
  change is minor, just confirm the update and point to the new image. Avoid
  repeating the entire component description multiple times.

---

## 4 — Modifying a component

### 4.1 Adjusting parameters

The simplest modification is changing the factory-function arguments. Always
call `gf.clear_cache()` before regenerating to avoid stale data.

### 4.2 Composing components

Build a custom component by placing and connecting sub-components. Use the
`@gf.cell` decorator for proper naming and caching.

---

## 5 — Exporting the component

```python
# Write to GDS file
gdspath = c.write_gds("/tmp/my_component.gds")
```

---

## 6 — Proactive Design Loop (Best Practices)

When the user asks for a component, follow this streamlined loop:

1. **Understand & Execute:** Identify the component and parameters. If the
   user asks for a sequence of steps, execute them as a batch where logical.
2. **Generate & Visualize:** Create the component and render the PNG.
3. **Show & Tell:** Share the image and a *short* summary of what changed.
4. **Anticipate:** If the next step is obvious, offer to perform it or just
   do it and show the result.
5. **Iterate Concisely:** For small tweaks, don't repeat the full initial
   explanation. Just show the new image and highlight the specific change.

---

## 7 — Common component quick-reference

| Component | Factory function | Key parameters |
|---|---|---|
| Straight waveguide | `gf.components.straight` | `length`, `width` |
| Euler bend | `gf.components.bend_euler` | `radius`, `angle` |
| 1×2 MMI | `gf.components.mmi1x2` | `width_mmi`, `length_mmi` |
| Ring resonator | `gf.components.ring_single` | `gap`, `radius` |
| Grating coupler | `gf.components.grating_coupler_te` | `period`, `n_periods` |

---

## 8 — When you are unsure: consult the docs and samples

The full gdsfactory docs are at **<https://gdsfactory.github.io/gdsfactory/>**.
Browse tutorial notebooks under `docs/notebooks/` or over 100 sample Python
scripts under `gdsfactory/samples/` for worked examples.

**Don't guess – search the repo for examples first.**

---

## 9 — KLayout API: advanced transformations with `DCplxTrans`

gdsfactory's geometry backend is [kfactory](https://github.com/gdsfactory/kfactory/),
which is itself built on top of KLayout's Python database (db) module.  When you
need fine-grained control over how component references are placed, you can
use the KLayout API directly through kfactory.

### 9.1 KLayout DB module reference

Full API docs: **<https://www.klayout.de/doc-qt5/code/module_db.html>**

### 9.2 `DCplxTrans` — rotation, mirroring and translation in one object

[`klayout.db.DCplxTrans`](https://www.klayout.de/doc-qt5/code/class_DCplxTrans.html)
represents a *complex transformation* (scale + rotate + mirror + translate)
operating on floating-point (micrometer) coordinates.  It is the most
expressive single transformation type in KLayout and covers every placement
operation you are likely to need.

Constructor signature (kwargs supported since recent KLayout versions):
```python
DCplxTrans(mag=1.0, angle=0.0, mirror=False, u=DVector(0, 0))
# mag    – uniform scaling factor (float). ALWAYS use 1.0 — see warning below.
# angle  – counter-clockwise rotation in degrees (float)
# mirror – mirror about the x-axis (applied first, before rotation)
# u      – translation as a DVector(dx, dy) in micrometers
# Application order: mirror → rotate → translate
```

> ⚠️ **Never set `mag` to anything other than `1.0`.**  No semiconductor
> foundry accepts layouts with a scaling factor applied to instances.
> Setting `mag != 1.0` causes DRC failures, incorrect critical dimensions,
> and subtle off-grid snapping issues (`dcplx_trans` is snapped to the
> manufacturing grid, so a non-unity mag will silently distort your
> geometry).  If you need to size a component differently, create a
> correctly-sized variant using the component's factory parameters instead.

### 9.3 Using `DCplxTrans` with gdsfactory via `ComponentReference.dcplx_trans`

Every `ComponentReference` in gdsfactory exposes a `.dcplx_trans` property
that reads and writes the underlying `DCplxTrans` transformation.  Setting it
is the most direct way to position a reference:

```python
import gdsfactory as gf
import klayout.db as kdb

gf.gpdk.PDK.activate()

circuit = gf.Component("circuit")
mmi_ref = circuit.add_ref(gf.components.mmi1x2())

# Place the reference: rotate 45 °, mirror, then translate to (10, 5) µm
# mag is always 1.0 — never use a different scaling factor
mmi_ref.dcplx_trans = kdb.DCplxTrans(
    1.0,                        # mag: always 1.0
    45.0,                       # angle: 45 ° CCW
    True,                       # mirror about x-axis
    kdb.DVector(10.0, 5.0),     # translate to (10, 5) µm
)
```

### 9.4 Combining transformations with the `*` operator

`DCplxTrans` objects can be **composed** with the `*` operator.  The result is
a single transformation that applies the right-hand operand first, then the
left-hand operand — just like matrix multiplication.

```python
# Build up a transformation incrementally (mag always 1.0)
rotate      = kdb.DCplxTrans(1.0, 90.0, False)             # rotate 90 °
translation = kdb.DCplxTrans(1.0, 0.0, False, kdb.DVector(20.0, 0.0))

# Combined: first rotate, then translate
combined = translation * rotate   # right-to-left application order

ref = circuit.add_ref(gf.components.straight())
ref.dcplx_trans = combined
```

You can also multiply a `DCplxTrans` by a `DPoint` or `DVector` to transform a
single coordinate:

```python
point = kdb.DPoint(1.0, 0.0)
transformed = combined * point   # returns a DPoint
```

### 9.5 Quick-reference: common `DCplxTrans` patterns

| Goal | Example |
|---|---|
| Translate only | `kdb.DCplxTrans(1.0, 0.0, False, kdb.DVector(dx, dy))` |
| Rotate 90 ° CCW | `kdb.DCplxTrans(1.0, 90.0, False)` |
| Mirror about x-axis | `kdb.DCplxTrans(1.0, 0.0, True)` |
| Rotate then translate | `translation * rotate` (right-to-left) |
| Read current transform | `ref.dcplx_trans` |
| Modify in place | `ref.dcplx_trans = new_trans * ref.dcplx_trans` |

> ℹ️ Note: `ComponentReference.dcplx_trans` snaps the transformation to the
> manufacturing grid internally (converting through `ICplxTrans`), so
> off-grid placements will be silently adjusted.  Always place references
> on-grid to avoid unexpected shifts.
