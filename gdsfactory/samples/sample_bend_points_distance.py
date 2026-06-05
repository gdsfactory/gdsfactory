"""Set bend point spacing globally via PDK.bend_points_distance.

PDK.bend_points_distance controls the default npoints calculation.
Smaller values produce denser point spacing.
The formula is: npoints = abs(angle) / 360 * radius / bend_points_distance / 2

Note: Set bend_points_distance before creating any bends, as cell caching
means the PDK value is captured at cell creation time.
"""

from __future__ import annotations

import gdsfactory as gf

gf.gpdk.PDK.activate()


if __name__ == "__main__":
    PDK = gf.get_active_pdk()
    radius = 10.0

    # Set coarser spacing (100 nm) before creating bends
    PDK.bend_points_distance = 0.1
    b = gf.components.bend_euler(radius=radius, angle=90)
    first_layer = next(iter(b.get_polygons()))
    n = b.get_polygons()[first_layer][0].num_points()
    print(f"bend_points_distance=100nm: {n} polygon points")

    # Reset to default
    PDK.bend_points_distance = 20e-3

    b.show()
