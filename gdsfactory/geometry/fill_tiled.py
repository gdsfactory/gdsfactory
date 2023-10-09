try:
    import kfactory as kf
    from kfactory.utils.fill import fill_tiled
except ImportError as e:
    print(
        "You can install `pip install gdsfactory[kfactory]` for using maskprep. "
        "And make sure you use python >= 3.10"
    )
    raise e

__all__ = ["fill_tiled"]


if __name__ == "__main__":
    import gdsfactory as gf
    import gdsfactory.geometry.fill_tiled as fill

    c = kf.KCell("ToFill")
    c.shapes(kf.kcl.layer(1, 0)).insert(
        kf.kdb.DPolygon.ellipse(kf.kdb.DBox(5000, 3000), 512)
    )
    c.shapes(kf.kcl.layer(10, 0)).insert(
        kf.kdb.DPolygon(
            [kf.kdb.DPoint(0, 0), kf.kdb.DPoint(5000, 0), kf.kdb.DPoint(5000, 3000)]
        )
    )

    fc = kf.KCell("fill")
    fc.shapes(fc.kcl.layer(2, 0)).insert(kf.kdb.DBox(20, 40))
    fc.shapes(fc.kcl.layer(3, 0)).insert(kf.kdb.DBox(30, 15))

    # fill.fill_tiled(c, fc, [(kf.kcl.layer(1,0), 0)], exclude_layers = [(kf.kcl.layer(10,0), 100), (kf.kcl.layer(2,0), 0), (kf.kcl.layer(3,0),0)], x_space=5, y_space=5)
    fill.fill_tiled(
        c,
        fc,
        [(kf.kcl.layer(1, 0), 0)],
        exclude_layers=[
            (kf.kcl.layer(10, 0), 100),
            (kf.kcl.layer(2, 0), 0),
            (kf.kcl.layer(3, 0), 0),
        ],
        x_space=5,
        y_space=5,
    )

    gdspath = "mzi_fill.gds"
    c.write(gdspath)
    gf.show(gdspath)
