"""Group references. Distribute them ...
"""

if __name__ == "__main__":

    import gdsfactory

    D = gdsfactory.Component()

    t1 = D << gdsfactory.components.text("1")
    t2 = D << gdsfactory.components.text("2")
    t3 = D << gdsfactory.components.text("3")
    t4 = D << gdsfactory.components.text("4")
    t5 = D << gdsfactory.components.text("5")
    t6 = D << gdsfactory.components.text("6")

    D.distribute(direction="x", spacing=3)

    gdsfactory.show(D)
