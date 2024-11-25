"""Lets pack a doe and export it with metadata."""

from __future__ import annotations

import gdsfactory as gf

if __name__ == "__main__":
    doe = gf.components.pack_doe(
        gf.components.mmi1x2,
        settings={"length_taper": (10, 15, 20, 30)},
        function="add_fiber_array",
    )
    doe.show()
    doe.write_gds(f"{__file__[:-3]}/test.gds")
