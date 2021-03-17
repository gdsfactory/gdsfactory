"""Group references. Distribute them ...
"""

if __name__ == "__main__":

    import pp

    D = pp.Component()

    t1 = D << pp.components.text("1")
    t2 = D << pp.components.text("2")
    t3 = D << pp.components.text("3")
    t4 = D << pp.components.text("4")
    t5 = D << pp.components.text("5")
    t6 = D << pp.components.text("6")

    D.distribute(direction="x", spacing=3)

    pp.show(D)
