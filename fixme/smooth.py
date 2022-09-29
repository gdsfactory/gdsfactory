"""fixme."""


if __name__ == "__main__":
    import gdsfactory as gf
    import numpy as np

    points = np.array([(20, 10), (40, 10), (20, 40), (50, 40), (50, 20), (70, 20)])

    p = gf.path.smooth(
        points=points,
        radius=2,
        bend=gf.path.euler,
        use_eff=False,
    )

    c = p.extrude(layer=(1, 0), width=0.1)
    c.show()
