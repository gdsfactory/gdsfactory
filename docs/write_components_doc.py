import pathlib
import gdsfactory as gf


filepath = pathlib.Path("components.rst")

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
]


with open(filepath, "w+") as f:
    f.write(
        """
Components
=============================
"""
    )

    for name in sorted(gf.c.factory.keys()):
        if name in skip or name.startswith("_"):
            continue
        print(name)
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

  c = gf.components.{name}()
  c.plot()

"""
            )
