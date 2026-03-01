import gdsfactory as gf
from gdsfactory.components import bend_s_offset
from gdsfactory.gpdk import PDK

if __name__ == "__main__":
    PDK.activate()
    c = gf.Component()

    s_bend = c << bend_s_offset(
        offset=127 / 2 - 3.2 / 2,
        width=1.200,
        with_euler=False,
        radius=50,
    )

    c.show()
