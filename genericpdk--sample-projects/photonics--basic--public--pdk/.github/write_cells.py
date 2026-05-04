"""Write docs."""

import inspect

from mypdk import _cells as cells
from mypdk.config import PATH

filepath = PATH.repo / "docs" / "cells.rst"

skip = {}

skip_plot: tuple[str, ...] = ("",)
skip_settings: tuple[str, ...] = ()


with open(filepath, "w+") as f:
    f.write(
        """

Cells
=============================
"""
    )

    for name in sorted(cells.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(cells[name])
        kwargs = ", ".join(
            [
                f"{p}={repr(sig.parameters[p].default)}"
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

.. autofunction:: mypdk.cells.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: mypdk.cells.{name}

.. plot::
  :include-source:

  from mypdk import PDK, cells

  PDK.activate()
  c = cells.{name}({kwargs}).copy()
  c.draw_ports()
  c.plot()

"""
            )
