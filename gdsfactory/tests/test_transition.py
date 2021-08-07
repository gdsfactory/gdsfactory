import pytest
from phidl.path import transition

import gdsfactory as gf


def test_transition():
    with pytest.raises(ValueError):

        path = gf.Path()
        path.append(gf.path.arc(radius=10, angle=90))
        path.append(gf.path.straight(length=10))
        path.append(gf.path.euler(radius=3, angle=-90))
        path.append(gf.path.straight(length=40))
        path.append(gf.path.arc(radius=8, angle=-45))
        path.append(gf.path.straight(length=10))
        path.append(gf.path.arc(radius=8, angle=45))
        path.append(gf.path.straight(length=10))

        X = gf.CrossSection()
        X.add(width=1, offset=0, layer=0)

        x2 = gf.CrossSection()
        x2.add(width=2, offset=0, layer=0)

        transition(X, x2)


if __name__ == "__main__":
    test_transition()
