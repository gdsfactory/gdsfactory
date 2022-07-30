import inspect
import pathlib

import gdsfactory as gf
from gdsfactory.serialization import clean_value_json


filepath = pathlib.Path(__file__).parent.absolute() / "components.rst"

skip = {
    "LIBRARY",
    "circuit_names",
    "component_factory",
    "component_names",
    "container_names",
    "component_names_test_ports",
    "component_names_skip_test",
    "component_names_skip_test_ports",
    "dataclasses",
    "library",
    "waveguide_template",
}

skip_plot = [
    "component_lattice",
    "component_sequence",
    "extend_port",
    "extend_ports_list",
]

skip_settings = {"vias"}


with open(filepath, "w+") as f:
    f.write(
        """

You can customize the Gdsfactory generic PDK Pcells for your fab and use it as an inspiration to build your own.


Components
=============================
"""
    )

    for name in sorted(gf.components.cells.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
        sig = inspect.signature(gf.components.cells[name])
        kwargs = ", ".join(
            [
                f"{p}={repr(clean_value_json(sig.parameters[p].default))}"
                for p in sig.parameters
                if isinstance(sig.parameters[p].default, (int, float, str, tuple))
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
