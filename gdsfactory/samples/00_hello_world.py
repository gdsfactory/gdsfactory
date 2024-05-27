import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component()
    ref1 = c.add_ref(gf.components.rectangle(size=(10, 10), layer=(1, 0)))
    ref2 = c.add_ref(gf.components.text("Hello", size=10, layer=(2, 0)))
    ref3 = c.add_ref(gf.components.text("world", size=10, layer=(2, 0)))

    ref1.dxmax = ref2.dxmin - 5
    ref3.dxmin = ref2.dxmax + 2
    ref3.drotate(30)
    c.show()
