"""Tests for the AWG length_increment feature."""

from __future__ import annotations

import numpy as np

from gdsfactory.components.filters.awg import awg


def test_awg_default_builds() -> None:
    c = awg()
    names = [p.name for p in c.ports]
    assert "o1" in names
    assert "arm_lengths" not in c.info  # default keeps original behavior


def test_awg_constant_length_increment() -> None:
    """Each arm is exactly length_increment longer than the previous one."""
    dl = 12.5
    c = awg(arms=8, length_increment=dl)
    lengths = np.asarray(c.info["arm_lengths"])
    assert np.allclose(np.diff(lengths), dl, atol=1e-3), lengths
