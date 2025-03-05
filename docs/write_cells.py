import functools
import inspect
import os

from gdsfactory.config import PATH
from gdsfactory.get_factories import get_cells
from gdsfactory.serialization import clean_value_json

components = PATH.module / "components"
filepath = PATH.repo / "docs" / "components.rst"

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

with open(filepath, "w+") as f:
    f.write(
        """

PCells
=============================

Parametric Cells for the Generic PDK.

Consider them a foundation from which you can draw inspiration. Feel free to modify their cross-sections and layers to tailor a unique PDK suited for any foundry of your choice.

By doing so, you'll possess a versatile, retargetable PDK, empowering you to design your circuits with speed and flexibility.

"""
    )

    for root, _dirs, files in sorted(os.walk(components)):
        if "__init__.py" not in files:
            continue

        folder_name = os.path.basename(root)
        f.write(f"\n\n{folder_name}\n=============================\n")

        # Dynamically import cells from each folder
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
            # Skip if the name is in the skip list or starts with "_"
            if name in skip or name.startswith("_"):
                continue

            # Get the cell function or object
            cell = cells[name]

            # Skip if it's an instance of functools.partial
            if skip_partials and isinstance(cell, functools.partial):
                continue

            sig = inspect.signature(cell)

            kwargs = ", ".join(
                [
                    f"{p}={clean_value_json(sig.parameters[p].default)!r}"
                    for p in sig.parameters
                    if isinstance(sig.parameters[p].default, int | float | str | tuple)
                    and p not in skip_settings
                ]
            )
            if name in skip_plot:
                f.write(
                    f"""


.. autofunction:: {module_path}.{name}

"""
                )
            else:
                f.write(
                    f"""


.. autofunction:: {module_path}.{name}

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.{name}({kwargs}).copy()
  c.draw_ports()
  c.plot()

"""
                )
