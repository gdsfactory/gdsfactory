"""
circle = gf.components.circle(radius=r, angle_resolution=2, layer=(1, 0))

When the angular resolution is small enough, the arc is not very smooth

circle = gf.components.circle(radius=r, angle_resolution=2, layer=(1, 0))

When the angular resolution is small enough, the arc is not very smooth

https://github.com/gdsfactory/gdsfactory/issues/72
"""


import gdsfactory as gf


def option1():
    r = 0.1
    c1 = gf.components.circle(radius=r, angle_resolution=10, layer=(1, 0))
    c1.show()


def option2():
    """not recommended"""
    r = 0.1
    c2 = gf.components.circle(radius=r, angle_resolution=2, layer=(1, 0))
    gdspath1 = c2.write_gds(
        precision=1e-9
    )  # 1nm is the default precision for most Photonics fabs
    gf.show(gdspath1)

    gdspath2 = c2.write_gds(
        precision=10e-12
    )  # you can also increase to 10pm resolution
    gf.show(gdspath2)


if __name__ == "__main__":
    option2()
