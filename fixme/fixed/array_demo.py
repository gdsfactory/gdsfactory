"""Array is now faster.

Using port.id
100 | 0.56

After removing port.id
100 | 0.24
"""

if __name__ == "__main__":
    import gdsfactory as gf
    import time

    start = time.time()
    n = 100

    c = gf.Component()
    base = gf.components.rectangle(size=(1.0, 1.0), layer="WG")
    c = gf.components.array(
        component=base, spacing=(2.0, 2.0), columns=n, rows=n, add_ports=True
    )
    end = time.time()
    print(end - start)
    c.show(show_ports=True)
