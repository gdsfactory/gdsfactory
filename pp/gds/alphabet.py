import pp

characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#()+,-./:;<=>?@[{|}~_"


@pp.cell
def alphabet(dx=10):
    c = pp.Component()
    x = 0
    for s in characters:
        ci = pp.c.text(text=s)
        ci.name = s
        char = c << ci.flatten()
        char.x = x
        x += dx

    return c


if __name__ == "__main__":
    c = alphabet()
    pp.write_gds(c, "alphabet.gds")
    c.show()
