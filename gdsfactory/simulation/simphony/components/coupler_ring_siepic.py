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
    from simphony.library import siepic

    from gdsfactory.simulationsimphony import plot_model

    c = siepic.ebeam_dc_halfring_straight(
        gap=200e-9, radius=12e-6, width=500e-9, thickness=220e-9, couple_length=0.0
    )
    plot_model(c)
    print(c)
    plt.show()
