import inspect
import pathlib

import gdsfactory as gf


filepath = pathlib.Path(__file__).parent.absolute() / "components.rst"

skip = {}

skip_plot = [
    "component_lattice",
    "component_sequence",
    "extend_port",
    "extend_ports_list",
]

skip_settings = {"vias"}

cell_names = sorted(list(gf.components.cells.keys())[:2])

with open(filepath, "w+") as f:
    f.write(
        """

Cells
=============================

.. currentmodule:: gdsfactory.components

.. autosummary::
   :toctree: _autosummary/

"""
    )

    for name in sorted(cell_names):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(gf.components.cells[name])
        kwargs = ", ".join(
            [
                f"{p}={repr(sig.parameters[p].default)}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, (int, float, str, tuple))
                and p not in skip_settings
            ]
        )
        f.write(f"   {name}\n")
