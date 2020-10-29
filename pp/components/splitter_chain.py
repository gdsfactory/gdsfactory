from typing import Callable
from pp.components.mmi1x2 import mmi1x2
import pp


@pp.autoname
def splitter_chain(component: Callable = mmi1x2, n_devices: int = 3, **kwargs):
    """ Chain of splitters

    .. code::

                __
             __|
          __|  |__
        _|  |__
         |__


    .. plot::
      :include-source:

      import pp

      c = pp.c.splitter_chain(component=pp.c.mmi1x2(), n_devices=3)
      pp.plotgds(c)

    """
    c = pp.Component()
    component = pp.call_if_func(component, **kwargs)
    cref = c.add_ref(component)

    bend = pp.c.bezier()
    c.add_port(name="W0", port=cref.ports["W0"])
    c.add_port(name="E0", port=cref.ports["E0"])

    for i in range(1, n_devices):
        bref = c.add_ref(bend)
        bref.connect(port="1", destination=cref.ports["E1"])

        component = pp.call_if_func(component)
        cref = c.add_ref(component)

        cref.connect(port="W0", destination=bref.ports["0"])
        c.add_port(name="E{}".format(i), port=cref.ports["E0"])

    c.add_port(name="E{}".format(n_devices), port=cref.ports["E1"])
    return c


if __name__ == "__main__":
    c = splitter_chain(component=pp.c.mmi1x2, n_devices=4)
    pp.show(c)
