import gdsfactory

characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#()+,-./:;<=>?@[{|}~_"


@gdsfactory.cell
def alphabet(dx=10):
    c = gdsfactory.Component()
    x = 0
    for s in characters:
        ci = gdsfactory.components.text(text=s)
        ci.name = s
        char = c << ci.flatten()
        char.x = x
        x += dx

    return c


if __name__ == "__main__":
    c = alphabet()
    c.write_gds("alphabet.gds")
    c.show()
