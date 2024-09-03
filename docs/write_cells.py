import functools
import inspect

import gdsfactory as gf
from gdsfactory.config import PATH
from gdsfactory.serialization import clean_value_json

filepath = PATH.repo / "docs" / "components.rst"

skip = {
    "grating_coupler_elliptical_te",
    "grating_coupler_elliptical_tm",
    "grating_coupler_te",
    "grating_coupler_tm",
    "mzi_arms",
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

    for name in sorted(gf.components.cells.keys()):
        # Skip if the name is in the skip list or starts with "_"
        if name in skip or name.startswith("_"):
            continue

        # Get the cell function or object
        cell = gf.components.cells[name]

        # Skip if it's an instance of functools.partial
        if skip_partials and isinstance(cell, functools.partial):
            continue

        print(name)
        sig = inspect.signature(cell)

        kwargs = ", ".join(
            [
                f"{p}={repr(clean_value_json(sig.parameters[p].default))}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, int | float | str | tuple)
                and p not in skip_settings
            ]
        )
        if name in skip_plot:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: gdsfactory.components.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: gdsfactory.components.{name}

.. plot::
  :include-source:

  import gdsfactory as gf

  c = gf.components.{name}({kwargs})
  c.plot()

"""
            )
