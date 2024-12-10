"""based on phidl tutorial.

# Connecting devices with connect()

The connect command allows you to connect ComponentReference ports together like Lego blocks.

"""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    c = gf.Component("straights_sample")

    wg1 = c << gf.components.straight(length=10, width=1)
    wg2 = c << gf.components.straight(length=10, width=2)
    wg3 = c << gf.components.straight(length=10, width=3)

    wg2.connect(
        port="o1", other=wg1["o2"], allow_width_mismatch=True, allow_layer_mismatch=True
    )
    wg3.connect(
        port="o1",
        other=wg2["o2"],
        allow_width_mismatch=True,
        allow_layer_mismatch=True,
    )

    c.show()  # show it in klayout
