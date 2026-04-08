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

## 9 — KLayout API: transformations with `DCplxTrans`

gdsfactory's geometry backend is [kfactory](https://github.com/gdsfactory/kfactory/), built on KLayout's Python `db` module.  Full API docs: <https://www.klayout.de/doc-qt5/code/module_db.html>

### 9.1 `DCplxTrans` constructor

[`klayout.db.DCplxTrans`](https://www.klayout.de/doc-qt5/code/class_DCplxTrans.html) represents a rotation + mirror + translation on µm-unit coordinates. Positional and keyword arguments are both supported:

```python
import klayout.db as kdb

# positional: DCplxTrans(mag, angle, mirror, u)
t = kdb.DCplxTrans(1.0, 45.0, True, kdb.DVector(10.0, 5.0))

# keyword (preferred for clarity):
t = kdb.DCplxTrans(mag=1.0, angle=45.0, mirror=True, u=kdb.DVector(10.0, 5.0))
```

Parameters (application order: mirror → rotate → translate):

| Parameter | Type | Description |
|---|---|---|
| `mag` | float | Scaling factor — **always `1.0`**; see warning below |
| `angle` | float | CCW rotation in degrees |
| `mirror` | bool | Mirror about x-axis before rotation |
| `u` | `DVector` | Translation in µm |

> ⚠️ **Never set `mag != 1.0`.** No foundry accepts scaled instances. Use component factory parameters to create differently-sized variants.

> ℹ️ `ComponentReference.dcplx_trans` snaps to the manufacturing grid via `ICplxTrans`, so off-grid placements are silently adjusted.

### 9.2 Setting `ComponentReference.dcplx_trans`

```python
import gdsfactory as gf
import klayout.db as kdb

circuit = gf.Component("circuit")
ref = circuit.add_ref(gf.components.mmi1x2())

# rotate 45°, mirror, translate to (10, 5) µm
ref.dcplx_trans = kdb.DCplxTrans(mag=1.0, angle=45.0, mirror=True, u=kdb.DVector(10.0, 5.0))
```

### 9.3 Composing transformations

`DCplxTrans` objects compose with `*` (right-hand operand applied first):

```python
rotate      = kdb.DCplxTrans(angle=90.0)
translation = kdb.DCplxTrans(u=kdb.DVector(20.0, 0.0))

ref.dcplx_trans = translation * rotate   # rotate first, then translate
```

### 9.4 Quick-reference

| Goal | Example |
|---|---|
| Translate | `kdb.DCplxTrans(u=kdb.DVector(dx, dy))` |
| Rotate 90° CCW | `kdb.DCplxTrans(angle=90.0)` |
| Mirror about x-axis | `kdb.DCplxTrans(mirror=True)` |
| Rotate then translate | `translation * rotate` |
| Read / modify in place | `ref.dcplx_trans` / `ref.dcplx_trans = t * ref.dcplx_trans` |
