r"""SIEPIC coupler sample.

.. code::

           n2            n4
           |             |
            \           /
             \         /
           ---=========---
        n1    length_x    n3

"""


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from simphony.libraries import siepic

    from gdsfactory.simulation.simphony import plot_model

    c = siepic.HalfRing(
        gap=200e-9, radius=12e-6, width=500e-9, thickness=220e-9, couple_length=0.0
    )
    c.rename_pins("n1", "n2", "n4", "n3")
    plot_model(c, pin_in="n1")
    print(c)
    plt.show()
