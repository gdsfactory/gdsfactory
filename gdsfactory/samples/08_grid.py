"""Group components in a cell using grid."""

if __name__ == "__main__":
    import gdsfactory as gf

    t1 = gf.components.text("1")
    t2 = gf.components.text("2")
    t3 = gf.components.text("3")
    t4 = gf.components.text("4")
    t5 = gf.components.text("5")
    t6 = gf.components.text("6")

    c = gf.grid([t1, t2, t3, t4, t5, t6])
    c.show()
