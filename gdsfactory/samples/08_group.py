"""Group references. Distribute them as you want.

Distribute different references in the X direction.

"""


if __name__ == "__main__":

    import gdsfactory as gf

    D = gf.Component()

    t1 = D << gf.components.text("1")
    t2 = D << gf.components.text("2")
    t3 = D << gf.components.text("3")
    t4 = D << gf.components.text("4")
    t5 = D << gf.components.text("5")
    t6 = D << gf.components.text("6")

    D.distribute(direction="x", spacing=3)
    D.show()
