from __future__ import annotations

import numpy as np

import gdsfactory as gf


def complicated_transition(t, width1, width2):
    """Complicated transition that grows, then shrinks."""
    wmax = 2 * width2
    widths = np.ones_like(t)
    widths[np.where(t <= 0.25)] = width1
    widths[np.where(np.logical_and(t > 0.25, t <= 0.5))] = width1 + (
        wmax - width1
    ) * np.linspace(0, 1, len(widths[np.where(np.logical_and(t > 0.25, t <= 0.5))]))
    widths[np.where(t > 0.5)] = wmax + (width2 - wmax) * np.linspace(
        0, 1, len(widths[np.where(t > 0.5)])
    )
    return widths


def test_transition_type_callable() -> None:
    width1 = 1.0
    width2 = 2.0
    x1 = gf.cross_section.strip(width=width1)
    x2 = gf.cross_section.strip(width=width2)

    xt = gf.path.transition(
        cross_section1=x1, cross_section2=x2, width_type=complicated_transition
    )
    straight = gf.components.straight(length=10, cross_section=xt, npoints=100)

    assert straight.ysize == 2 * width2


if __name__ == "__main__":
    test_transition_type_callable()

    width1 = 1.0
    width2 = 2.0
    x1 = gf.cross_section.strip(width=width1)
    x2 = gf.cross_section.strip(width=width2)

    xt = gf.path.transition(
        cross_section1=x1, cross_section2=x2, width_type=complicated_transition
    )
    straight = gf.components.straight(length=10, cross_section=xt, npoints=100)
    straight.show()
