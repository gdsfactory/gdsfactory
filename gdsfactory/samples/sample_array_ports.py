import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component("sample_array_ports")
    b = c << gf.components.bend_euler()
    s = c.add_ref(
        gf.components.straight(length=10),
        rows=2,
        row_pitch=20,
        columns=2,
        column_pitch=20,
    )
    b.connect("o1", s["o2", 1, 1])
    c.show()
