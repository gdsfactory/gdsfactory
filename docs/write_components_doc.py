import pathlib
from pp.components import component_factory


p = pathlib.Path("components.rst")

skip_plot = [
    "component_lattice",
    "component_sequence",
]

with open(p, "w+") as f:
    f.write(
        """
Components
=============================
"""
    )

    for name in sorted(list(component_factory.keys())):
        print(name)
        if name in skip_plot:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: pp.components.{name}

"""
            )
        else:
            f.write(
                f"""

{name}
----------------------------------------------------

.. autofunction:: pp.components.{name}

.. plot::
  :include-source:

  import pp

  c = pp.components.{name}()
  c.plot()

"""
            )
