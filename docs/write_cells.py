"""Generates `docs/components.md` with all gdsfactory default PDK cells.

- Walks through the `gdsfactory/components` directory
- Finds all component modules (subfolders with __init__.py)
- Extracts all cell functions from each module
- Generates Markdown with mkdocstrings directives and rendered component plots
- Writes to `docs/components.md`
"""

import inspect
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gdsfactory.config import PATH
from gdsfactory.get_factories import get_cells
from gdsfactory.serialization import clean_value_json

components = PATH.module / "components"
filepath = PATH.repo / "docs" / "components.md"
img_dir = PATH.repo / "docs" / "components_images"
img_dir.mkdir(exist_ok=True)

skip = {
    "bbox",
    "grating_coupler_elliptical_te",
    "grating_coupler_elliptical_tm",
    "grating_coupler_te",
    "grating_coupler_tm",
    "mzi1x2",
    "mzi2x2_2x2",
    "mzi_coupler",
    "via1",
    "via2",
    "viac",
    "via_stack_heater_m2",
    "via_stack_heater_m3",
    "via_stack_heater_mtop",
    "via_stack_slab_m3",
    "via_stack_slot_m1_m2",
    "bend_euler180",
    "bend_circular180",
    "pack",
    "pack_doe",
    "pack_doe_grid",
}

skip_plot = [
    "component_lattice",
    "component_sequence",
    "extend_port",
    "extend_ports_list",
]

skip_settings = {"vias"}
skip_partials = False

import gdsfactory as gf

gf.gpdk.PDK.activate()

with open(filepath, "w+", encoding="utf-8") as f:
    f.write(
        """# PCells

Parametric Cells for the Generic PDK.

Consider them a foundation from which you can draw inspiration. Feel free to modify their cross-sections and layers to tailor a unique PDK suited for any foundry of your choice.

By doing so, you'll possess a versatile, retargetable PDK, empowering you to design your circuits with speed and flexibility.

"""
    )

    for root, _dirs, files in sorted(os.walk(components)):
        if "__init__.py" not in files:
            continue

        folder_name = os.path.basename(root)
        f.write(f"\n## {folder_name}\n\n")

        folder_path = root.replace(str(PATH.module), "gdsfactory")
        module_path = folder_path.replace(os.sep, ".")

        if module_path == "gdsfactory.components":
            continue

        try:
            module = __import__(module_path, fromlist=["__init__"])
            cells = get_cells([module])
            print(f"Imported module {module_path}, with {len(cells)} cells")
        except Exception as e:
            print(f"Error importing module {module_path}: {e}")
            continue

        for name in sorted(cells.keys()):
            if name in skip or name.startswith("_"):
                continue

            cell = cells[name]
            sig = inspect.signature(cell)

            kwargs = ", ".join(
                [
                    f"{p}={clean_value_json(sig.parameters[p].default)!r}"
                    for p in sig.parameters
                    if isinstance(sig.parameters[p].default, int | float | str | tuple)
                    and p not in skip_settings
                ]
            )

            f.write(f"::: {module_path}.{name}\n\n")

            if name not in skip_plot:
                img_path = img_dir / f"{name}.png"
                try:
                    c = gf.components.__getattr__(name)().copy()
                    c.draw_ports()
                    fig = c.plot(return_fig=True)
                    fig.savefig(str(img_path), bbox_inches="tight", pad_inches=0, dpi=80)
                    plt.close(fig)
                    f.write(f"![{name}](components_images/{name}.png)\n\n")
                    print(f"  Plotted {name}")
                except Exception as e:
                    print(f"  Error plotting {name}: {e}")
                    f.write(
                        f"""```python
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.components.{name}({kwargs}).copy()
c.draw_ports()
c.plot()
```

"""
                    )
