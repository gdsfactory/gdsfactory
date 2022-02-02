import gdsfactory as gf


if __name__ == "__main__":
    c1 = gf.components.straight(length=5)
    c2 = gf.components.straight(length=5.0)

    print(c1.name)
    print(c2.name)

    c = gf.Component()
    r1 = c << c1
    r2 = c << c2
    r2.movey(3)
    c.show()
