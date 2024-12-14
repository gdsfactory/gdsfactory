"""# Components. You can adapt some component functions from the `gdsfactory.components` module. Each function there returns a Component object. Here are two equivalent functions."""

from __future__ import annotations

from functools import partial
from typing import Any

import gdsfactory as gf


def straight_wide1(width: float = 10, **kwargs: Any) -> gf.Component:
    return gf.components.straight(width=width, **kwargs)


straight_wide2 = partial(gf.components.straight, width=10)


if __name__ == "__main__":
    # c = straight_wide1()
    c = straight_wide2()
    c.show()
