"""Write docs."""

import inspect

from gpdk import PDK
from gpdk.config import PATH

filepath = PATH.repo / "docs" / "components.rst"

skip = {}

skip_plot: tuple[str, ...] = ("",)
skip_settings: tuple[str, ...] = ()
cells = PDK.cells


with open(filepath, "w+") as f:
    f.write(
        """

Cells generic pdk
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

.. autofunction:: gpdk.components.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: gpdk.{name}

.. plot::
  :include-source:

  from gpdk import PDK, components

  PDK.activate()
  c = components.{name}({kwargs}).copy()
  c.draw_ports()
  c.plot()

"""
            )
