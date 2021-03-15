from pp.components import component_factory
import pathlib


p = pathlib.Path("components.rst")

with open(p, "w+") as f:
    f.write(
        """
Components
=============================
"""
    )

    for name in sorted(list(component_factory.keys())):
        print(name)
        f.write(
            f"""

{name}
----------------------------------------------------

.. autofunction:: pp.c.{name}

.. plot::
  :include-source:

  import pp

  c = pp.c.{name}()
  c.plot()

"""
        )
