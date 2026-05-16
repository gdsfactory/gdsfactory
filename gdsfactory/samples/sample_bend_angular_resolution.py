"""Compare bend angular resolutions.

Shows the effect of different angular_step values on bend point density.
Smaller angular_step = smoother curves with more points.
"""

from __future__ import annotations

import gdsfactory as gf

gf.gpdk.PDK.activate()


if __name__ == "__main__":
    c = gf.Component()
    radius = 10.0
    y = 0.0

    for angular_step in [1, 5, 15, 30]:
        bend = gf.components.bend_euler(
            radius=radius, angle=90, angular_step=angular_step
        )
        ref = c << bend
        ref.movey(y)
        label = f"angular_step={angular_step}deg"
        text = c << gf.components.text(label, size=2, layer="TEXT")
        text.move((15, y - 2))
        y += 20

    c.show()
