from __future__ import annotations

import gdsfactory as gf
from gdsfactory.config import rich_output


def test_rich_output_top_level_export() -> None:
    assert gf.rich_output is rich_output
